import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .utils import tprint, finder


class Spike:
    def __init__(self, path, coord=None, angle=0.0, rotation=0.0, rotation_holder=0.0):
        """
        Initialize the Spike object.

        Parameters:
        -----------
        path : str
            The path to the spike file.
        coord : list or numpy array, optional (default: None)
            The coordinates of the probe in the brain from bregma in mm.
            * (AP, DV, ML)
            * AP: anterior-posterior from the bregma in mm (positive: anterior)
            * DV: dorsal-ventral from the pial surface in mm (positive: ventral)
            * ML: medial-lateral from the bregma in mm (positive: right)

            * If you are using wide probe like Neuropixels 2.0, the coord is the center of the probe.

        angle: float, optional (default: 0.0)
            The angle of the probe in the brain in degrees (positive: anterior).
            = ~pitch of the manipulator
            * 0 degrees indicates the probe is pointing posterior (vertical).
            * positive values indicate the probe is pointing anterior.
            * negative values indicate the probe is pointing posterior.

        rotation: float, optional (default: 0.0)
            The rotation of the manipulator in the brain in degrees (positive: clockwise).
            = ~yaw of the manipulator
            * 0 degrees indicates the manipulator arm is facing posterior.
            * positive values indicate the manipulator arm is facing left (you rotated the manipulator clockwise).
            * negative values indicate the manipulator arm is facing right (you rotated the manipulator counter-clockwise).

        rotation_holder: float, optional (default: 0.0)
            The rotation of the probe in the holder in degrees (positive: clockwise).
            = ~yaw of the probe in the holder
            * 0 degrees indicates the probe is facing posterior.
            * positive values indicate the probe is facing left (you rotated the probe holder clockwise).
            * negative values indicate the probe is facing right (you rotated the probe holder counter-clockwise).

        """
        self.path = path

        tprint(f"Loading {self.path}")
        data = {
            k.lower(): pd.DataFrame(v) if isinstance(v, (list, np.ndarray)) else v
            for k, v in loadmat(path, simplify_cells=True).items()
            if not k.startswith("__")
        }
        self.__dict__.update(data)

        if coord is not None:
            if len(coord) != 3:
                raise ValueError("coord must be a list or numpy array with 3 elements")

            self.coord = np.array(coord)
            self.angle = np.radians(angle)
            self.rotation = np.radians(rotation)
            self.rotation_holder = np.radians(rotation_holder)
            self.load_atlas()
            self.load_channel_position()
            self.load_unit_position()

    def __repr__(self):
        """
        Return a string representation of the Spike object structure
        """
        result = []
        for key, value in self.__dict__.items():
            if key.startswith("__"):
                continue
            if isinstance(value, pd.DataFrame):
                result.append(key)
                for col in value.columns:
                    result.append(f"    {col}: {value[col].shape}")
            elif isinstance(value, dict):
                result.append(key)
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        result.append(f"    {k}: {v.shape}")
                    elif isinstance(v, list):
                        result.append(f"    {k}: {len(v)}")
                    elif isinstance(v, dict):
                        result.append(f"    {k}:")
                    elif isinstance(v, (int, float)):
                        result.append(f"    {k}: {v}")
            elif isinstance(value, str):
                result.append(f"{key}: {value}")
            elif isinstance(value, np.ndarray):
                result.append(f"{key}: {value.shape}")
        return "\n".join(result)

    @property
    def spike_df(self):
        """
        Return a dataframe of spike time and unit id.
        """
        spk = np.concatenate(self.spike["time"])
        idx = np.concatenate(
            [np.full_like(x, i) for i, x in enumerate(self.spike["time"])]
        )
        spkidx = np.argsort(spk)
        return pd.DataFrame({"time": spk[spkidx], "unit": idx[spkidx]})

    def spike_bin(self, bin_size=0.10, window_size=None):
        """
        Return an array of binned spikes by time and unit id for decoding analysis.

        * If window_size is None, return the binned spikes without convolution.
        * window_size should be odd number.
        """
        spk = self.spike_df
        xedges = np.arange(spk.time.min(), spk.time.max(), bin_size)
        yedges = np.arange(spk.unit.max() + 2)
        xcenter = xedges[:-1] + bin_size / 2

        spks, _, _ = np.histogram2d(
            spk.time.values, spk.unit.values, bins=(xedges, yedges)
        )

        if window_size is None:
            return xcenter, yedges[:-1], spks

        assert window_size % 2 == 1, "window_size must be odd"
        spks_conv = smooth(spks, type="boxcar", sigma=window_size, mode="valid", axis=0)

        return xcenter[window_size // 2 : -window_size // 2 + 1], yedges[:-1], spks_conv

    @property
    def time_nidq(self):
        if not hasattr(self, "nidq"):
            return None

        chans = np.unique(self.nidq.chan.values)
        types = np.unique(self.nidq.type.values)

        time_nidq = {}
        for chan in chans:
            for type in types:
                temp = self.nidq.query(f"chan == {chan} and type == {type}")
                if temp.empty:
                    continue
                time_nidq[f"{chan}_{type}"] = temp["time_imec"].values

        return time_nidq

    def load_atlas(self):
        """Load brain atlas and initialize scene"""
        tprint("Loading brain atlas")
        import vedo

        vedo.settings.default_backend = "vtk"

        from brainrender import Scene

        self.scene = Scene()
        self.atlas = self.scene.atlas
        self.origin = self._coord_to_origin()

    def _coord_to_origin(self):
        """Convert coordinates in mm to indices in um."""
        # Constants
        BREGMA = np.array([5400, 0, 5700])  # in um
        SCALING = np.array([-1000, 1000, -1000])

        # Convert to atlas space
        coord_surface = self.coord * SCALING + BREGMA

        # Get surface index
        resolution = np.array(self.atlas.resolution)
        idx = np.round(coord_surface / resolution).astype(int)
        surface_idx = np.where(self.atlas.annotation[idx[0], :, idx[2]] != 0)[0][0]

        # Calculate surface coordinates
        dv_surface = surface_idx * resolution[1]
        ap_surface = coord_surface[0]
        ml_surface = coord_surface[2]

        # Calculate lengths
        length_surface_to_tip = self.coord[1] * 1000
        length_channel_to_tip = self.spike.get("meta", {}).get("imTipLength", 0)
        length_surface_to_channel = length_surface_to_tip - length_channel_to_tip

        # Calculate offsets
        dv_surface_to_channel = length_surface_to_channel * np.cos(self.angle)
        length_horizontal = length_surface_to_channel * np.sin(self.angle)
        ap_surface_to_channel = length_horizontal * np.cos(self.rotation)
        ml_surface_to_channel = length_horizontal * np.sin(self.rotation)

        # Calculate final channel coordinates
        return np.array(
            [
                ap_surface - ap_surface_to_channel,
                dv_surface + dv_surface_to_channel,
                ml_surface + ml_surface_to_channel,
            ]
        )

    def load_channel_position(self):
        """Load channel locations"""
        tprint("Loading channel locations")
        meta = self.spike.get("meta", {})
        snsGeomMap = meta.get("snsGeomMap", {})
        electrodes = snsGeomMap.get("electrodes")

        if not electrodes:
            return

        df = pd.DataFrame(electrodes)
        shank_spacing = snsGeomMap.get("header", {}).get("shank_spacing", 0)

        # Calculate x positions with shank offset
        channel_position_x = df["x"].values + shank_spacing * df["shank"].values
        # Center the x positions
        self.position_x_correction = (
            channel_position_x.min() + channel_position_x.max()
        ) // 2
        channel_position_x -= self.position_x_correction
        channel_position_z = df["z"].values

        # Create position array and calculate coordinates
        self.channel_position = np.column_stack(
            (channel_position_x, channel_position_z)
        )
        self.channel_coords = self._position_to_coords(self.channel_position)
        self.channel_region = [
            self.atlas.structure_from_coords(c, microns=True, as_acronym=True)
            for c in self.channel_coords
        ]

    def _position_to_coords(self, position):
        """Convert probe positions to atlas coordinates"""
        shape_um = self.atlas.shape_um
        resolution = np.array(self.atlas.resolution)

        # x, z = position
        pos = np.column_stack(
            (np.zeros_like(position[:, 0]), position[:, 1], position[:, 0])
        ).T  # (3, 384): (AP, DV, ML)G

        # Rotation of the probe in the holder
        R_holder = np.array(
            [
                [np.cos(self.rotation_holder), 0, -np.sin(self.rotation_holder)],
                [0, 1, 0],
                [np.sin(self.rotation_holder), 0, np.cos(self.rotation_holder)],
            ]
        )

        R_angle = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle), 0],
                [np.sin(self.angle), np.cos(self.angle), 0],
                [0, 0, 1],
            ]
        )

        R_rotation = np.array(
            [
                [np.cos(self.rotation), 0, -np.sin(self.rotation)],
                [0, 1, 0],
                [np.sin(self.rotation), 0, np.cos(self.rotation)],
            ]
        )

        pos_holder = R_holder @ pos
        pos_angle = R_angle @ pos_holder
        pos_rotation = R_rotation @ pos_angle

        coords = pos_rotation.T * np.array([-1, -1, 1]) + self.origin
        return np.clip(coords, 0, shape_um - resolution)

    def load_unit_position(self):
        """Load unit locations"""
        tprint("Loading unit locations")
        self.unit_position = self.spike["waveform_position"][:, 0, :]
        self.unit_position[:, 0] -= self.position_x_correction
        self.unit_coords = self._position_to_coords(self.unit_position)
        self.unit_region = [
            self.atlas.structure_from_coords(c, microns=True, as_acronym=True)
            for c in self.unit_coords
        ]

    def add_region(self, region, alpha=0.2, hemisphere="both"):
        """Add brain region"""
        self.scene.add_brain_region(region, alpha=alpha, hemisphere=hemisphere)

    def plot_brain(self):
        """Plot brain with channel and unit positions"""
        from brainrender.actors import Points, Line

        self.scene.add(Points(self.channel_coords, colors="yellow", alpha=0.2))
        self.scene.add(Points(self.unit_coords, colors="red", alpha=0.5, radius=50))
        self.scene.render()

    def plot_probe(self):
        """Plot probe and unit location"""
        unit_position = self.spike["waveform_position"][:, 0, :]
        channel_position = self.spike["channel_position"]

        plt.figure(figsize=(4, 10))
        plt.scatter(
            channel_position[:, 0],
            channel_position[:, 1],
            s=12,
            c="gray",
            marker="s",
            alpha=0.2,
        )
        plt.scatter(
            unit_position[:, 0],
            unit_position[:, 1],
            s=20,
            c="red",
            marker="o",
            alpha=0.5,
        )

        range_min = channel_position.min(axis=0)
        range_max = channel_position.max(axis=0)
        plt.xlim(range_min[0] - 24, range_max[0] + 24)
        plt.ylim(range_min[1] - 24, range_max[1] + 24)

        plt.show()

    def plot(self, event=None):
        import tkinter as tk
        from tkinter import ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        if event is None:
            event = self.time_nidq

        if event is None:
            print("No event found")
            return

        if not isinstance(event, dict):
            event = {"event": event}

        trial_type = list(event.keys())
        unit_ids = list(range(self.spike["time"].shape[0]))

        def update_plot():
            selected_type = frame.type_var.get()
            selected_unit = int(frame.unit_var.get())
            window = [float(frame.window_start.get()), float(frame.window_end.get())]
            reorder = int(frame.reorder_var.get())
            bin_size = float(frame.bin_size_var.get())
            sigma = int(frame.sigma_var.get())

            time_spike = self.spike["time"][selected_unit]
            time_trial = event[selected_type]

            plot_raster_psth(
                time_spike,
                time_trial,
                window=window,
                reorder=reorder,
                bin_size=bin_size,
                sigma=sigma,
                fig=fig,
            )
            canvas.draw()

        root = tk.Tk()
        root.title("Spike Plot GUI")

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        params = [
            ("Trial Type:", "type_var", trial_type[0], trial_type),
            ("Unit ID:", "unit_var", "0", unit_ids),
            ("Window Start:", "window_start", "-5", None),
            ("Window End:", "window_end", "5", None),
            ("Reorder:", "reorder_var", "1", None),
            ("Bin Size:", "bin_size_var", "0.01", None),
            ("Sigma:", "sigma_var", "10", None),
        ]

        for label, var_name, default, values in params:
            ttk.Label(frame, text=label).grid(
                column=0, row=frame.grid_size()[1], sticky=tk.W
            )
            var = tk.StringVar(value=default)
            if values is not None:
                widget = ttk.Combobox(frame, textvariable=var, values=values)
            else:
                widget = ttk.Entry(frame, textvariable=var)
            widget.grid(column=1, row=frame.grid_size()[1] - 1, sticky=(tk.W, tk.E))
            setattr(frame, var_name, var)

        ttk.Button(frame, text="Update Plot", command=update_plot).grid(
            column=1, row=frame.grid_size()[1], sticky=tk.E
        )

        fig = plt.Figure(figsize=(10, 8))
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().grid(row=1, column=0)

        update_plot()
        root.mainloop()


def align(time_spike, time_event, window=[-5, 5]):
    time_aligned = [
        time_spike[(time_spike >= te + window[0]) & (time_spike <= te + window[1])] - te
        if not np.isnan(te)
        else np.array([])
        for te in time_event
    ]
    return np.array(time_aligned, dtype=object)


def count_spike(time_spike, time_event, window=[-0.5, 0.5]):
    time_spike = np.asarray(time_spike)
    time_event = np.asarray(time_event)

    orig_shape = time_event.shape
    flat_event = time_event.ravel()

    idx_start = np.searchsorted(time_spike, flat_event + window[0], side="left")
    idx_end = np.searchsorted(time_spike, flat_event + window[1], side="right")

    counts = (idx_end - idx_start).astype(float)
    counts[np.isnan(flat_event)] = np.nan

    return counts.reshape(orig_shape)


def get_raster(time_aligned, type_event, reorder=1, line=True):
    if type_event.dtype == float:
        type_event = type_event.astype(int)
    n_type = np.max(type_event) + 1
    n_trial_type = np.bincount(type_event)
    cum_trial = np.concatenate([0, np.cumsum(n_trial_type)], axis=None)

    x = np.empty(n_type, dtype=object)
    y = np.empty(n_type, dtype=object)

    if time_aligned.size == 0:
        return x, y

    n_spike = np.vectorize(len)(time_aligned)
    for i_type in range(n_type):
        in_trial = type_event == i_type
        n_spike_type = n_spike[in_trial]
        x_temp = np.concatenate(time_aligned[in_trial])
        y_temp = (
            np.repeat(
                np.arange(1, n_trial_type[i_type] + 1, dtype=float)
                if reorder
                else np.arange(1, len(type_event) + 1, dtype=float)[in_trial],
                n_spike_type,
            )
            + cum_trial[i_type]
        )

        if line:
            x[i_type] = np.column_stack(
                (x_temp, x_temp, np.full_like(x_temp, np.nan))
            ).ravel()
            y[i_type] = np.column_stack(
                (y_temp, y_temp + 1, np.full_like(y_temp, np.nan))
            ).ravel()
        else:
            x[i_type] = x_temp
            y[i_type] = y_temp

    return x, y


def get_spike_bin(time_aligned, bin_size=0.01, window=[-5, 5]):
    bins = np.arange(window[0] - bin_size / 2, window[1] + bin_size, bin_size)
    t = bins[:-1] + bin_size / 2
    n_trial = len(time_aligned)

    time_binned = np.zeros((n_trial, len(t)))
    for i_trial, time_trial in enumerate(time_aligned):
        time_binned[i_trial], _ = np.histogram(time_trial, bins)

    return t, time_binned


def smooth(time_binned, type="gaussian", sigma=10, axis=1, mode="same"):
    if sigma == 0:
        return time_binned

    if type == "gaussian":
        if hasattr(signal, "gaussian"):
            window = signal.gaussian(5 * sigma, sigma)
        else:
            window = signal.windows.gaussian(5 * sigma, sigma)
        window /= np.sum(window)
    elif type == "boxcar":
        window = np.ones(sigma)
    else:
        raise ValueError(f"Invalid smoothing type: {type}")

    time_binned_conv = np.apply_along_axis(
        lambda m: np.convolve(m, window, mode=mode), axis=axis, arr=time_binned
    )
    return time_binned_conv


def get_psth(
    time_aligned, type_event, bin_size=0.01, sigma=10, window=[-5, 5], do_smooth=True
):
    t, time_binned = get_spike_bin(time_aligned, bin_size, window)
    time_conv = smooth(time_binned, sigma=sigma) if do_smooth else time_binned

    types, type_counts = np.unique(type_event, return_counts=True)
    n_type = len(types)

    psth = np.zeros((n_type, len(t)))
    psth_sem = np.zeros((n_type, len(t)))
    for i, (i_type, count) in enumerate(zip(types, type_counts)):
        in_trial = type_event == i_type
        psth[i] = np.nansum(time_conv[in_trial], axis=0) / (bin_size * count)
        psth_sem[i] = np.nanstd(time_conv[in_trial], axis=0) / (
            bin_size * np.sqrt(count)
        )

    return t, psth, psth_sem


def get_raster_psth(
    time_spike,
    time_event,
    type_event=None,
    window=[-5, 5],
    reorder=1,
    bin_size=0.01,
    sigma=10,
    line=True,
):
    if type_event is None:
        type_event = np.zeros(len(time_event))

    in_event = ~np.isnan(type_event)
    time_event = time_event[in_event]
    type_event = type_event[in_event].astype("int64")
    type_unique, type_index = np.unique(type_event, return_inverse=True)

    time_aligned = align(time_spike, time_event, window)
    x, y = get_raster(time_aligned, type_index, reorder, line)
    t, psth, psth_sem = get_psth(time_aligned, type_index, bin_size, sigma, window)

    return {"x": x, "y": y}, {"t": t, "y": psth, "sem": psth_sem}


def plot_raster_psth(
    time_spike,
    time_event,
    type_event=None,
    window=[-5, 5],
    reorder=1,
    bin_size=0.01,
    sigma=10,
    fig=None,
    plot_bar=False,
):
    """
    Plot raster and PSTH (Peri-Stimulus Time Histogram) for spike data.

    Parameters:
    -----------
    time_spike : array-like
        Spike times.
    time_event : array-like
        Event times to align spikes to.
    type_event : array-like, optional
        Event types for different conditions. If None, all events are treated as the same type.
        * It can be the array of event types such as [0, 0, 1, 0, 2, 1, ...].
        * The length of type_event should be the same as time_event.
    window : list of two floats, default [-5, 5]
        Time window around each event for analysis, in seconds.
    reorder : int, default 1
        If 1, raster plot will be ordered by type_event. If 0, original order is maintained.
    bin_size : float, default 0.01
        Size of time bins for PSTH, in seconds.
    sigma : int, default 10
        Smoothing parameter for PSTH.
        If 0, no smoothing is applied.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure is created.
    plot_bar : bool, default False
        If True, plot PSTH as a bar plot instead of a line plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the raster and PSTH plots.
    """
    if fig is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    else:
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

    window_raw = np.array(window) + np.array(
        [-bin_size * sigma * 3, bin_size * sigma * 3]
    )
    raster, psth = get_raster_psth(
        time_spike,
        time_event,
        type_event,
        window=window_raw,
        reorder=reorder,
        bin_size=bin_size,
        sigma=sigma,
    )

    cmap = [(0, 0, 0)] + list(plt.get_cmap("tab10").colors)
    for i, (x, y, y_psth, y_sem) in enumerate(
        zip(raster["x"], raster["y"], psth["y"], psth["sem"])
    ):
        color = cmap[i % len(cmap)]
        if x is not None and y is not None:
            ax1.plot(x, y, color=color, linewidth=0.5)

        if y_psth is not None:
            if not plot_bar:
                ax2.plot(psth["t"], y_psth, color=color)
                ax2.fill_between(
                    psth["t"],
                    y_psth - y_sem,
                    y_psth + y_sem,
                    color=color,
                    alpha=0.2,
                    linewidth=0,
                )
            else:
                ax2.bar(psth["t"], y_psth, color=color, width=bin_size, linewidth=0)

    ylim_raster = [0, max((np.nanmax(y) if y is not None else 1) for y in raster["y"])]
    ylim_psth = [0, np.nanmax(psth["y"] + psth["sem"]) * 1.1]

    for ax, ylim in [(ax1, ylim_raster), (ax2, ylim_psth)]:
        ax.vlines(0, 0, ylim[1], color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlim(window)
        ax.set_ylim(ylim)
        ax.spines[["top", "right"]].set_visible(False)

    ax1.set_ylabel("Trial")
    ax2.set_ylabel("Firing Rate (Hz)")
    ax2.set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_tagging(
    time_spikes, time_onset, time_offset=None, window=[-0.05, 0.1], bin_size=0.001
):
    """
    Plot raster, PSTH, and cumulative plot for spike tagging analysis.

    Parameters:
    -----------
    time_spikes : list of array-like
        List of spike times for each unit.
    time_onset : array-like
        Onset times of the tagging events.
    time_offset : array-like, optional
        Offset times of the tagging events. If None, assumed to be the same as time_onset.
    window : list of two floats, default [-0.05, 0.1]
        Time window around each event for analysis, in seconds.
    bin_size : float, default 0.001
        Size of time bins for PSTH, in seconds.
    """
    n_unit = len(time_spikes)

    if time_offset is None:
        time_offset = time_onset

    cmap = [(0, 0, 0)] + list(plt.get_cmap("tab10").colors)
    f = plt.figure(figsize=(10, 4 * n_unit))
    gs_main = gridspec.GridSpec(n_unit, 2, wspace=0.3, hspace=0.3)

    ylim_raster = [0, len(time_onset)]
    ylim_c = [0, 1]

    for i_unit in range(n_unit):
        # calculate raster and psth
        raster, psth = get_raster_psth(
            time_spikes[i_unit], time_onset, window=window, bin_size=bin_size, sigma=0
        )
        ylim_psth = [0, np.nanmax(psth["y"]) * 1.1]

        # calculate cumulative plot
        c = get_latency(
            time_spikes[i_unit], time_onset, time_offset, duration=window[1]
        )

        gs_unit = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_main[i_unit, 0], hspace=0.05
        )
        ax0 = f.add_subplot(gs_unit[0])
        ax1 = f.add_subplot(gs_unit[1])

        gs_latency = gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=gs_main[i_unit, 1]
        )
        ax2 = f.add_subplot(gs_latency[0])

        for i, (x, y, y_psth) in enumerate(zip(raster["x"], raster["y"], psth["y"])):
            color = cmap[i % len(cmap)]
            if x is not None and y is not None:
                ax0.plot(x, y, color=color, linewidth=0.5)

            if y_psth is not None:
                ax1.bar(psth["t"], y_psth, color=color, width=bin_size, linewidth=0)

        ax2.plot(
            c["time_event"], c["count_event"], color=[0.0, 0.718, 1.0], linewidth=0.75
        )
        ax2.plot(
            c["time_base"], c["count_base"], color="gray", linestyle="--", linewidth=0.5
        )

        # Set labels and titles
        ax0.set_title(f"Unit {i_unit + 1}")
        ax0.set_ylabel("Trial")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Firing Rate (Hz)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Cumulative Probability")

        # Set x-axis limits for raster and PSTH
        ax0.set_xlim(window)
        ax1.set_xlim(window)
        ax2.set_xlim([0, window[1]])
        ax0.set_ylim(ylim_raster)
        ax1.set_ylim(ylim_psth)
        ax2.set_ylim(ylim_c)

        # Remove top and right spines
        for ax in [ax0, ax1, ax2]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Add vertical line at t=0
        for ax in [ax0, ax1]:
            ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.5)


def get_latency(spike, event_onset, event_offset, duration=0.08, offset=0.3):
    assert len(event_onset) == len(event_offset)
    n_event = len(event_onset)

    spike_event = [
        spike[(spike >= event_onset[0]) & (spike < event_onset[0] + duration)]
        - event_onset[0]
    ]
    spike_base = []
    for i_event in range(n_event - 1):
        base = np.arange(
            event_offset[i_event] + offset, event_onset[i_event + 1], duration
        )
        n_base_temp = len(base) - 1
        if n_base_temp < 1:
            continue

        spike_event.append(
            spike[
                (spike >= event_onset[i_event + 1])
                & (spike < event_onset[i_event + 1] + duration)
            ]
            - event_onset[i_event + 1]
        )

        base_index = np.random.permutation(np.arange(n_base_temp))[
            : min(n_base_temp, 50)
        ]
        for i_base in base_index:
            spike_base.append(
                spike[(spike >= base[i_base]) & (spike < base[i_base + 1])]
                - base[i_base]
            )

    spike_event_all = np.sort(np.concatenate(spike_event))
    count_event = np.arange(len(spike_event_all)) / len(spike_event_all)
    spike_base_all = np.sort(np.concatenate(spike_base))
    count_base = np.arange(len(spike_base_all)) / len(spike_base_all)

    return {
        "time_event": spike_event_all,
        "count_event": count_event,
        "time_base": spike_base_all,
        "count_base": count_base,
    }


if __name__ == "__main__":
    path = finder(
        path=r"Z:\kimd\project-stroke\data_ephys",
        msg="Select a session file",
        pattern=r".mat$",
    )
    spike = Spike(path, [1, 1.5, -1.75], 0, 0, -90)
    spike.add_region("VTA", hemisphere="left")
    spike.plot_brain()
    print(spike.unit_region)

    # # interactive plot
    # spike.plot()

    # plot raster and psth
    # time_event = spike.nidq.query('chan == 5 and type == 1')['time_imec'].values
    # plot_tagging(spike.spike['time'], time_event)
    # plt.show()
    breakpoint()

