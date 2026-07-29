import os
import datetime

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

from .utils import tprint, finder, savemat_safe

VOLT_PER_BIT = 10 / 32768  # 10 V / int16 (2^15)


def to_value(text):
    if text is None:
        return None
    s = text.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def parse_state(shard):
    """Parsing <PVStateValue> information."""
    label = lambda e, attr: (e.get("description") or "").strip() or e.get(attr)

    state = {}
    for sv in shard.findall("PVStateValue"):
        key = sv.get("key")
        if "value" in sv.attrib:
            state[key] = to_value(sv.get("value"))
            continue

        entry = {
            label(iv, "index"): to_value(iv.get("value"))
            for iv in sv.findall("IndexedValue")
        }
        for group in sv.findall("SubindexedValues"):
            entry[group.get("index")] = {
                label(siv, "subindex"): to_value(siv.get("value"))
                for siv in group.findall("SubindexedValue")
            }
        state[key] = entry
    return state


def parse_xml(elem):
    """Parsing a XML file into a dictionary."""
    if not len(elem):
        return to_value(elem.text)

    out = {}
    for child in elem:
        value = parse_xml(child)
        if child.tag in out:
            if not isinstance(out[child.tag], list):
                out[child.tag] = [out[child.tag]]
            out[child.tag].append(value)
        else:
            out[child.tag] = value
    return out


class Ophys:
    """A PrairieView imaging session."""

    def __init__(self, path=None, path_vrec=None):
        self.path = path
        self.fn_env = finder(path=path, pattern=r"\.env$")
        if not self.fn_env:
            raise FileNotFoundError(f"No .env file found under {path}")

        self.fn_xml = self.fn_env.replace(".env", ".xml")
        if not os.path.exists(self.fn_xml):
            raise FileNotFoundError(f"XML file not found at {self.fn_xml}")
        self.fn_ome = self.fn_env.replace(".env", ".companion.ome")

        self.meta = None
        self.frames = None
        self.ome = None
        self.vrec = None

        self.n_frame_xml = None
        self.n_frame_ome = None
        self.n_frame_tiff = None

        self.load_xml()
        if os.path.exists(self.fn_ome):
            self.load_ome()
        self.count_tiff()

    def __repr__(self):
        trim = lambda v: round(v, 6) if isinstance(v, float) else v

        result = [f"Ophys: {os.path.basename(self.fn_xml)}"]
        for key in (
            "laserPower",
            "pmtGain",
            "framePeriod",
            "rastersPerFrame",
            "linesPerFrame",
            "pixelsPerLine",
            "opticalZoom",
        ):
            value = (self.meta or {}).get(key)
            if isinstance(value, dict):
                result.append(f"    {key}:")
                result += [f"        {k}: {trim(v)}" for k, v in value.items()]
            elif value is not None:
                result.append(f"    {key}: {trim(value)}")

        if self.frames is not None:
            result.append(f"frames: {self.frames.shape} {list(self.frames.columns)}")
            result.append(
                f"    {self.frames.cycle.nunique()} cycle(s), "
                f"{self.frames.channel.nunique()} channel(s), "
                f"{self.frame_rate:.2f} Hz"
            )
        if self.ome is not None:
            result.append(f"ome: {self.ome.shape} {list(self.ome.columns)}")

        count = {
            k: v
            for k, v in (
                ("xml", self.n_frame_xml),
                ("ome", self.n_frame_ome),
                ("tiff", self.n_frame_tiff),
            )
            if v is not None
        }
        if count:
            line = "n_frame: " + ", ".join(f"{k} {v}" for k, v in count.items())
            result.append(
                line
                if len(set(count.values())) == 1
                else f"\033[91m{line}  MISMATCH\033[0m"
            )
        return "\n".join(result)

    def load_xml(self):
        """
        Read metadata and the frame table
        """
        tprint(f"Loading {os.path.basename(self.fn_xml)}")

        meta, meta_done, n_done = {}, False, 0
        cycle, index, abs_t, rel_t, filename, page, channel = ([] for _ in range(7))

        for _, elem in ET.iterparse(self.fn_xml, events=("end",)):
            if elem.tag == "Frame":
                for file in elem.iterfind("File"):
                    index.append(elem.get("index"))
                    abs_t.append(elem.get("absoluteTime"))
                    rel_t.append(elem.get("relativeTime"))
                    filename.append(file.get("filename"))
                    page.append(file.get("page"))
                    channel.append(file.get("channel"))
                elem.clear()
            elif elem.tag == "Sequence":
                self.type = elem.get("type")
                self.time = datetime.time.fromisoformat(elem.get("time"))
                cycle.extend([elem.get("cycle")] * (len(index) - n_done))
                n_done = len(index)
                elem.clear()
            elif elem.tag == "PVStateShard" and not meta_done:
                meta, meta_done = parse_state(elem), True
                elem.clear()

        self.meta = meta
        self.time
        self.frames = pd.DataFrame(
            {
                "cycle": np.array(cycle, dtype=np.int32),
                "index": np.array(index, dtype=np.int64),
                "absoluteTime": np.array(abs_t, dtype=np.float64),
                "relativeTime": np.array(rel_t, dtype=np.float64),
                # A few multi-page OME-TIFFs back tens of thousands of frames.
                "filename": pd.Categorical(filename),
                "page": np.array(page, dtype=np.int64),
                "channel": np.array(channel, dtype=np.int16),
            }
        )
        self.n_frame_xml = len(self.frames)

        tprint(
            f"Loaded {len(self.frames)} frames "
            f"({self.frames.cycle.nunique()} cycle(s), "
            f"{self.frames.channel.nunique()} channel(s))"
        )
        return self.meta, self.frames

    def load_ome(self):
        """
        Per-plane table from the companion OME-XML.

        <TiffData> maps a plane to its file and IFD (= page - 1, IFD being
        0-based); <Plane> carries the timestamp and stage position, which the
        PrairieView XML does not hold per frame. Both appear once per plane in
        the same order, so the two runs line up row for row.
        """
        tprint(f"Loading {os.path.basename(self.fn_ome)}")

        filename, ifd, t, z, c, delta_t, pos_x, pos_y, pos_z = ([] for _ in range(9))
        for _, elem in ET.iterparse(self.fn_ome, events=("end",)):
            if elem.tag.endswith("TiffData"):
                uuid = next(iter(elem), None)  # <UUID FileName="..."> child
                filename.append(uuid.get("FileName") if uuid is not None else None)
                ifd.append(elem.get("IFD"))
                elem.clear()
            elif elem.tag.endswith("Plane"):
                t.append(elem.get("TheT"))
                z.append(elem.get("TheZ"))
                c.append(elem.get("TheC"))
                delta_t.append(elem.get("DeltaT"))
                pos_x.append(elem.get("PositionX"))
                pos_y.append(elem.get("PositionY"))
                pos_z.append(elem.get("PositionZ"))
                elem.clear()

        self.ome = pd.DataFrame(
            {
                "t": np.array(t, dtype=np.int64),
                "z": np.array(z, dtype=np.int32),
                "c": np.array(c, dtype=np.int16),
                "deltaTime": np.array(delta_t, dtype=np.float64),
                "positionX": np.array(pos_x, dtype=np.float64),
                "positionY": np.array(pos_y, dtype=np.float64),
                "positionZ": np.array(pos_z, dtype=np.float64),
                "filename": pd.Categorical(filename),
                "ifd": np.array(ifd, dtype=np.int64),
            }
        )
        self.n_frame_ome = len(self.ome)

    def count_tiff(self, exact=False):
        """
        Total pages across the OME-TIFFs of the session.
        """
        if not exact and os.path.exists(self.fn_ome):
            for _, elem in ET.iterparse(self.fn_ome, events=("start",)):
                # Stop at <Pixels>; the 35k <TiffData> entries below it would
                # cost as much to parse as the IFD walk this is avoiding.
                if elem.tag.endswith("Pixels"):
                    self.n_frame_tiff = (
                        int(elem.get("SizeT"))
                        * int(elem.get("SizeZ"))
                        * int(elem.get("SizeC"))
                    )
                    return

        import tifffile

        folder = os.path.dirname(self.fn_xml)
        count = 0
        for fn in self.frames.filename.unique():
            with tifffile.TiffFile(os.path.join(folder, fn)) as tif:
                count += len(tif.pages)
        self.n_frame_tiff = count

    @property
    def frame_rate(self):
        """Measured frame rate (Hz); extra channels repeat a frame's timestamp."""
        frames = self.frames.drop_duplicates(["cycle", "index"])
        # Never difference across the gap between cycles.
        return 1 / frames.groupby("cycle", observed=True).absoluteTime.diff().mean()


class VRec:
    """A reader for PrairieView voltage recording."""

    def __init__(self, path=None, task=None):
        self.path = path

        self.fn_xml = finder(path=path, pattern=r"_VoltageRecording_\d+\.xml$")
        if not self.fn_xml:
            raise FileNotFoundError(f"No VoltageRecording XML found under {path}")

        # Parse the XML
        self.meta = parse_xml(ET.parse(self.fn_xml).getroot())
        experiment = self.meta["Experiment"]

        signals = experiment["SignalList"]["VRecSignal"]
        if isinstance(signals, dict):
            signals = [signals]

        self.time = datetime.datetime.fromisoformat(self.meta["DateTime"])
        self.signals = [s for s in signals if s["Enabled"]]
        self.channels = [s["Name"] for s in self.signals]
        self.n_channel = len(self.channels)
        self.sample_rate = float(experiment["Rate"])
        self.n_sample = int(self.meta["SamplesAcquired"])
        self.fn_bin = os.path.join(os.path.dirname(self.fn_xml), self.meta["DataFile"])

        self.data_raw = None
        self.data = None
        self.trial = None

        # Load the binary data
        self.load_data()
        self.parse_data()

        # Parsing of the Unity task
        if task == "vr":
            self.parse_task(task)

    def __repr__(self):
        out = [
            f"VRec: {os.path.basename(self.fn_xml)}",
            f"{self.n_sample} samples @ {self.sample_rate:g} Hz = {self.duration:.1f} s",
        ]

        for channel, pulse in (self.data or {}).items():
            duration = (pulse[:, 1] - pulse[:, 0]).mean() if len(pulse) else 0
            out.append(f"    {channel}: {len(pulse)} pulses, duration: {duration:.3f} s")

        if self.trial:
            out.append(
                f"trial: {self.trial['nTrialNidq']} trials, "
                f"{self.trial['resultNidq'].sum()} rewarded"
            )
            out.append(f"    {list(self.trial)}")
        return "\n".join(out)

    @property
    def duration(self):
        """Recording length in seconds."""
        return self.n_sample / self.sample_rate

    def load_data(self):
        """Loading the binary data"""
        n_sample_raw = os.path.getsize(self.fn_bin) // 2 // self.n_channel
        if n_sample_raw != self.n_sample:
            tprint(
                f"{self.fn_bin} holds {n_sample_raw} samples for {self.n_channel} "
                f"channels, mismatched with {self.n_sample} samples in the XML"
            )

        tprint(
            f"Loading {os.path.basename(self.fn_bin)}: {self.n_sample} samples "
            f"@ {self.sample_rate:g} Hz = {self.duration:.1f} s, "
            f"{self.n_channel} channels {self.channels} "
            f"({n_sample_raw - self.n_sample} padding rows dropped)"
        )

        data = np.memmap(self.fn_bin, dtype="<i2", mode="r")
        self.data_raw = data[:n_sample_raw * self.n_channel].reshape(n_sample_raw, -1)[:self.n_sample] # remove padding rows

    def parse_data(self, threshold=2.5, jitter=0.002):
        """
        Pulse on/off times (s) per channel, as an (n_pulse, 2) array.

        self.data[channel] = np.stack([onsets, offsets], axis=1)

        Args:
            threshold: logic level (V) separating low from high.
            jitter: pulses shorter than this (s) are dropped as glitches.

        Notes:
            - `high` is padded with a low sample at both ends, so a line that is
              already high at the first sample (or still high at the last) still
              yields matching numbers of onsets and offsets.
        """
        self.trial = None
        self.t0 = 0.0

        data, dropped = {}, {}
        for c, channel in enumerate(self.channels):
            high = np.zeros(len(self.data_raw) + 2, dtype=bool)
            np.greater(self.data_raw[:, c], threshold / VOLT_PER_BIT, out=high[1:-1])

            on = np.flatnonzero(~high[:-1] & high[1:]) / self.sample_rate
            off = np.flatnonzero(high[:-1] & ~high[1:]) / self.sample_rate
            pulse = np.stack([on, off], axis=1)

            is_good = (pulse[:, 1] - pulse[:, 0]) >= jitter
            if not is_good.all():
                dropped[channel] = int((~is_good).sum())
            data[channel] = pulse[is_good]

        if dropped:
            tprint(
                f"Dropped pulses shorter than {jitter * 1e3:g} ms: "
                + ", ".join(f"{ch} x{n}" for ch, n in dropped.items())
            )

        # Realign by the first frame onset (AI 1) to match the PrairieView XML frame table
        if len(data.get("AI 1", ())):
            self.t0 = data["AI 1"][0, 0]
            for channel in self.channels:
                data[channel] -= self.t0

        self.data = data

    def frame2time(self, idx):
        """Convert a frame index to time (s) using the frame table."""
        if self.data is None:
            raise ValueError("Data are not loaded")

        time_frame = np.mean(self.data["AI 1"], axis=1) # midpoint of each frame
        return time_frame[idx]

    def time2frame(self, t):
        """
        Convert time (s) to the nearest frame index using the frame table.

        frame 0 --| frame 1 --| frame 2 --|... -- frame n-1
                       |
                       t --> (1)
        """
        if self.data is None:
            raise ValueError("Data are not loaded")

        return np.searchsorted(self.data["AI 1"][:, 0], t, side="right") - 1

    def parse_task(self, task="vr", threshold=2.5):
        """
        Trial structure from the task TTL channels.

        AI 1: 2-photon imaging frame onset (1) and offset (0)
        AI 2: task start (1) and end (0)
        AI 3: delay start (1) and delay end (0) and cue start
        AI 4: choice/ITI start (1) and next trial delay start (0)
        AI 5: left cue (AI2 0) / choice (AI4 1)
        AI 6: water reward
        """
        if task != "vr":
            raise ValueError(f"Unsupported task type: {task}")

        delay, choice, reward = (self.data[c] for c in ("AI 3", "AI 4", "AI 6"))
        n_trial = min(len(delay), len(choice))

        time_start, time_cue = delay[:n_trial, 0], delay[:n_trial, 1]
        time_choice, time_end = choice[:n_trial, 0], choice[:n_trial, 1]

        direction = self.channels.index("AI 5")
        level = lambda t: (
            np.asarray(self.data_raw[((t + self.t0) * self.sample_rate).astype(np.int64), direction])
            > threshold / VOLT_PER_BIT
        )
        cue = level((time_cue + time_choice) / 2) + 1 # 1: left, 2: right
        choice = level((time_choice + time_end) / 2) + 1 # 1: left, 2: right

        result_index = np.searchsorted(time_start, reward[:, 0], side="right") - 1
        in_task = (result_index >= 0) & (result_index < n_trial)
        result = np.bincount(result_index[in_task], minlength=n_trial)

        self.trial = {
            "nTrialNidq": n_trial,
            "timeStartNidq": time_start, # delay start
            "timeCueNidq": time_cue, # cue start
            "timeChoiceNidq": time_choice, # choice start (reward given)
            "timeEndNidq": time_end, # iti end (next trial delay start)
            "cueNidq": cue.astype(np.int8),
            "choiceNidq": choice.astype(np.int8),
            "resultNidq": result.astype(np.int8),
        }

        tprint(
            f"Parsed {n_trial} trials: {result.sum()} rewarded "
            f"({result.mean():.1%}), {cue.mean() - 1:.1%} left cues, {choice.mean() - 1:.1%} left choices"
        )
        return self.trial

    def save(self, path=None):
        """
        Save the VRec data to a MATLAB file.
        """
        if path is None:
            fd = os.path.dirname(self.fn_xml)
            path = finder(path=fd, folder=False, multiple=False, pattern=r'.mat$')
            if path is None:
                fn = os.path.basename(fd) + '_data.mat'
                print("No path provided. Saving .mat file in the current directory.")
                path = os.path.join(fd, fn)

        data = {key: getattr(self, key, None) for key in ('channels', 'data', 'trial')}
        data = {key: value for key, value in data.items() if value is not None}
        if not data:
            tprint("Nothing to save")
            return

        savemat_safe(path, data)

if __name__ == "__main__":
    fd = "/home/kimd/Downloads"

    vrec = VRec(fd, task="vr")
    print(vrec)

    # ophys = Ophys(fd)
    # print(ophys)

    breakpoint()
