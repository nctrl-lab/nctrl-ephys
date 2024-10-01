import os
import numpy as np
import pandas as pd

from nptdms import TdmsFile

from .utils import finder

def read_tdms(tdms_fn, threshold=2.5):
    with TdmsFile.open(tdms_fn) as tdms:
        analog = tdms['Analog']
        sample_rate = analog.properties['ScanRate']

        events = []
        for channel in analog:
            data = analog[channel].read_data() > threshold
            changes = np.where(np.diff(data) != 0)[0] + 1

            event_dict = {
                'time': changes / sample_rate,
                'frame': changes,
                'chan': int(''.join(filter(str.isdigit, channel))),
                'type': data[changes]
            }
            events.append(pd.DataFrame(event_dict))

    return pd.concat(events, ignore_index=True)

def save_tdms(tdms_fns=None,path=None):
    if tdms_fns is None:
        path = os.path.join(os.path.expanduser("~"), 'Documents', 'Measurement Computing', 'DAQami')
        tdms_fns = finder(path=path, pattern=r'\.tdms$', multiple=True)

    for tdms_fn in tdms_fns:
        fn = tdms_fn.replace('.tdms', '.pd')
        if not os.path.exists(fn):
            print(f"Saving {fn}")
            events = read_tdms(tdms_fn)
            events.to_pickle(fn)
        else:
            print(f"Skipping {fn} (already exists)")

if __name__ == "__main__":
    save_tdms()