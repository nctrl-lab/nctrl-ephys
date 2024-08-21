import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import finder
from .spikeglx import read_bin

class BMI:
    def __init__(self, path=None):
        path = finder(path, pattern='\.prb$', folder=True)

        if not path or not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist')

        self.path = path
        self.file_names = ['mua.bin', 'spk.bin', 'spk_wav.bin', 'fet.bin', os.path.join('spktag', 'model.pd')]
        self.file_paths = {fn: os.path.join(path, fn) for fn in self.file_names if os.path.exists(os.path.join(path, fn))}
        self.file_paths['prb'] = next((os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith('.prb')), None)

        self.prb_fn = self.file_paths.get('prb')
        self.mua_fn = self.file_paths.get('mua.bin')
        self.spk_fn = self.file_paths.get('spk.bin')
        self.spk_wav_fn = self.file_paths.get('spk_wav.bin')
        self.fet_fn = self.file_paths.get('fet.bin')
        self.model_fn = self.file_paths.get(os.path.join('spktag', 'model.pd'))

        print(f'  prb:     {self.prb_fn}\n'
              f'  mua:     {self.mua_fn}\n'
              f'  spk:     {self.spk_fn}\n'
              f'  spk_wav: {self.spk_wav_fn}\n'
              f'  fet:     {self.fet_fn}\n'
              f'  model:   {self.model_fn}')

        self.n_channel = 160
        self.sample_rate = 25000
        self.binary_radix = 13
        self.uV_per_bit = 0.195

        self.load_prb()

    def load_prb(self):
        if self.prb_fn:
            with open(self.prb_fn, 'r') as f:
                prb = json.load(f)
                self.n_channel = prb['params']['n_ch']
                self.sample_rate = prb['params']['fs']
                self.channel_id = np.zeros(self.n_channel, dtype=int)
                self.channel_position = np.full((self.n_channel, 2), np.nan)
                for i, (key, value) in enumerate(prb['pos'].items()):
                    self.channel_id[i] = int(key)
                    self.channel_position[int(key), :] = value
    
    def plot_prb(self):
        _, ax = plt.subplots(figsize=(6, 8))
        ax.scatter(self.channel_position[:, 0], self.channel_position[:, 1], alpha=0.6, marker='s', s=4)

        for i in range(self.n_channel):
            ax.annotate(self.channel_id[i], (self.channel_position[i, 0], self.channel_position[i, 1]), textcoords='offset points', xytext=(3, 0), ha='left', va='center', fontsize=8)
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
        # selected_channel = channel_idx
        self.binary_radix = 13
        self.uV_per_bit = 0.195
        data = read_bin(self.mua_fn, n_channel=self.n_channel, dtype='int32', channel_idx=selected_channel, sample_range=sample_range)

        return data / (2 ** self.binary_radix) * self.uV_per_bit
    
    def plot_mua(self, channel_idx=slice(0, 128), sample_range=(5000, 10000)):
        data = self.load_mua(channel_idx=channel_idx, sample_range=sample_range)

        data -= np.median(data, axis=0, keepdims=True)
        data -= np.median(data, axis=1, keepdims=True)

        for i in range(data.shape[1]):
            plt.plot(data[:, i] + i * 100, label=f'Channel {i}', color='k')  # Offset each channel for visibility
        plt.xlabel('Samples')
        plt.ylabel('Channel')
        plt.title('Raw data')
        plt.show()


if __name__ == '__main__':
    bmi = BMI('C:\\SGL_DATA')
    bmi.plot_mua()


