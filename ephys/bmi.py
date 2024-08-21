import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import finder
from .spikeglx import read_bin

class BMI:
    def __init__(self, path=None):
        path = finder(path, pattern=r'\.prb$', folder=True)

        if not path or not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist')

        self.path = path
        self.file_names = ['mua.bin', 'spk.bin', 'spk_wav.bin', 'fet.bin', os.path.join('spktag', 'model.pd')]
        self.file_paths = {fn: os.path.join(path, fn) for fn in self.file_names if os.path.exists(os.path.join(path, fn))}
        self.file_paths['prb'] = next((os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith('.prb')), None)

        for file_type in ['prb', 'mua.bin', 'spk.bin', 'spk_wav.bin', 'fet.bin', os.path.join('spktag', 'model.pd')]:
            setattr(self, f"{file_type.split('.')[0]}_fn", self.file_paths.get(file_type))

        print('\n'.join(f"  {k}: {v}" for k, v in self.file_paths.items()))

        self.n_channel = 160
        self.sample_rate = 25000.0
        self.binary_radix = 13
        self.uV_per_bit = 0.195

        self.load_prb()

    def load_prb(self):
        if self.prb_fn:
            with open(self.prb_fn, 'r') as f:
                prb = json.load(f)
                self.n_channel = prb['params']['n_ch']
                self.sample_rate = float(prb['params']['fs'])
                self.channel_id = np.arange(self.n_channel, dtype=int)
                self.channel_position = np.full((self.n_channel, 2), np.nan)
                for i, (key, value) in enumerate(prb['pos'].items()):
                    self.channel_id[i] = int(key)
                    self.channel_position[int(key), :] = value
    
    def plot_prb(self):
        _, ax = plt.subplots(figsize=(6, 8))
        ax.scatter(self.channel_position[:, 0], self.channel_position[:, 1], alpha=0.6, marker='s', s=4)

        for i in range(self.n_channel):
            ax.annotate(self.channel_id[i], (self.channel_position[i, 0], self.channel_position[i, 1]), 
                        textcoords='offset points', xytext=(3, 0), ha='left', va='center', fontsize=8)
        ax.set_xlabel('x position (um)')
        ax.set_ylabel('y position (um)')
        ax.set_title(f'Probe {self.prb_fn}')
        ax.axis('equal')
        plt.show()

    def load_mua(self, channel_idx=slice(0, 128), sample_range=None):
        """
        Load mua data from binary file.

        Check OneDrive/nclab/manual/ephys_intan/Intan_RHD2000_series_datasheet.pdf for more details.
        Amplifier Differential Gain: 192 V/V
        Amplifier AC Input Voltage Range: +/- 5 mV
        Voltage Step Size of ADC (Least Significant Bit): 0.195 uV
        """
        selected_channel = self.channel_id[channel_idx]
        data = read_bin(self.mua_fn, n_channel=self.n_channel, dtype='int32', 
                        channel_idx=selected_channel, sample_range=sample_range)

        return data / (2 ** self.binary_radix) * self.uV_per_bit
    
    def plot_mua(self, channel_idx=slice(0, 128), sample_range=(5000, 10000)):
        data = self.load_mua(channel_idx=channel_idx, sample_range=sample_range)

        data -= np.median(data, axis=0, keepdims=True)
        data -= np.median(data, axis=1, keepdims=True)

        plt.figure(figsize=(12, 8))
        for i in range(data.shape[1]):
            plt.plot(data[:, i] + i * 100, color='k', linewidth=0.5)
        plt.xlabel('Samples')
        plt.ylabel('Channel')
        plt.title('Raw data')
        plt.show()

    def load_spk(self):
        if self.spk_fn:
            spk = np.fromfile(self.spk_fn, dtype='<i4').reshape(-1, 2)
            self.spk_frame = spk[:, 0]
            self.spk_channel = spk[:, 1]  # Note: not in physical order
    
    def load_spk_wav(self):
        if self.spk_wav_fn:
            spk_wav = np.fromfile(self.spk_wav_fn, dtype=np.int32).reshape(-1, 20, 4)
            self.spk_peak_ch, self.spk_time, self.electrode_group = spk_wav[..., 0, 1], spk_wav[..., 0, 2], spk_wav[..., 0, 3]
            self.spk_wav = spk_wav[..., 1:, :] / (2**self.binary_radix) * self.uV_per_bit
    
    def load_fet(self):
        if self.fet_fn:
            scale_factor = float(2**self.binary_radix) / self.uV_per_bit

            fet = np.fromfile(self.fet_fn, dtype=np.int32).reshape(-1, 8)
            self.fet = pd.DataFrame({
                'frame': fet[:, 0].astype(np.int64),
                'group_id': fet[:, 1],
                'fet0': fet[:, 2].astype(np.float32) / scale_factor,
                'fet1': fet[:, 3].astype(np.float32) / scale_factor,
                'fet2': fet[:, 4].astype(np.float32) / scale_factor,
                'fet3': fet[:, 5].astype(np.float32) / scale_factor,
                'spike_id': fet[:, 6],
                'spike_energy': fet[:, 7].astype(np.float32) / scale_factor
            })

            # Check for any potential issues in the loaded data
            if self.fet['frame'].diff().min() < 0:
                print("\033[91mWarning: Time values are not monotonically increasing.\033[0m")
            if np.any(np.isnan(self.fet.values)):
                print("\033[91mWarning: NaN values detected in fet data.\033[0m")
            if np.any(np.isinf(self.fet.values)):
                print("\033[91mWarning: Infinite values detected in fet data.\033[0m")

            # TODO: If frame rolls over, add 2^32 to the frame


if __name__ == '__main__':
    bmi = BMI('C:\\SGL_DATA')
    bmi.load_spk()
    bmi.load_spk_wav()
    bmi.load_fet()
