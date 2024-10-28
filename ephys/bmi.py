import os
import re
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
import inquirer

from .utils import finder, tprint, file_reorder, sync
from .spikeglx import read_bin, read_digital
from .tdms import read_tdms

class BMI:
    def __init__(self, path=None, pattern=r'\.prb$'):
        """
        Initialize BMI class.

        Notes
        -----
        - The path to the .prb file is the main session folder.
        - It is assumed that there is only one .prb file in the session folder.
        - There can be multiple mua.bin, fet.bin, (nidq.bin or *.tdms) files in the session folder.
        - The number of mua.bin, fet.bin files should be the same.
        - Make sure to copy (nidq.bin or *.tdms) files under the subfolder such as main_path/nidq/**_g0/**_g0_t0.nidq.meta, main_path/nidq/**_g1/**_g1_t0.nidq.meta, etc.

            |-- mua.bin (1) --|    |-- mua.bin (2) --|    |-- mua.bin (3) --|
            |-- fet.bin (1) --|    |-- fet.bin (2) --|    |-- fet.bin (3) --|
                                  |--- nidq.bin (2) ---|
        
        - These files are going to be ordered by the time of the file creation.
        """
        self.time_sync_fpga = np.array([0, 1, 3, 64, 105, 181, 266, 284, 382, 469, 531,
            545, 551, 614, 712, 726, 810, 830, 846, 893, 983, 1024,
            1113, 1196, 1214, 1242, 1257, 1285, 1379, 1477, 1537, 1567, 1634,
            1697, 1718, 1744, 1749, 1811, 1862, 1917, 1995, 2047])  # in seconds
        self.n_channel = 160
        self.sample_rate = 25000.0
        self.binary_radix = 13
        self.uV_per_bit = 0.195

        path = finder(path, pattern=pattern, folder=True)
        if not path or not os.path.exists(path):
            raise ValueError(f'Path {path} does not exist')
        self.path = path
        self.session_name = os.path.basename(path)

        self.file_patterns = {
            'prb': r'\.prb$',
            'mua': r'mua\.bin$',
            'spk': r'spk\.bin$',
            'spk_wav': r'spk_wav\.bin$',
            'fet': r'fet\.bin$',
            'model': r'spktag[/\\]model\.pd$'
        }
        self.file_paths = {key: [] for key in self.file_patterns}

        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                for key, pattern in self.file_patterns.items():
                    if re.search(pattern, file_path):
                        self.file_paths[key].append(file_path)

        for file_type, paths in self.file_paths.items():
            setattr(self, f"{file_type}_fn", paths)
        
        # Order the file paths by the time of the file creation
        for file_type, paths in self.file_paths.items():
            self.file_paths[file_type] = sorted(paths, key=lambda x: os.path.getmtime(x))

        for file_type, paths in self.file_paths.items():
            print(f"  {file_type}:")
            for path in paths:
                print(f"    {path}")

        self.load_prb()

    def load_prb(self):
        if self.prb_fn:
            with open(self.prb_fn[0], 'r') as f:
                tprint(f"Loading {self.prb_fn[0]}")
                prb = json.load(f)
                self.n_channel = prb['params']['n_ch']
                self.sample_rate = float(prb['params']['fs'])
                self.channel_id = np.arange(self.n_channel, dtype=int)
                self.channel_position = np.full((self.n_channel, 2), np.nan)
                for i, (key, value) in enumerate(prb['pos'].items()):
                    self.channel_id[i] = int(key)
                    self.channel_position[i] = value
                
                self.channel_position[:, 0] -= np.nanmin(self.channel_position[:, 0])
                self.channel_position[:, 0] *= 2

                _, self.shank = np.unique(self.channel_position[:, 0], return_inverse=True)
                self.shank[np.isnan(self.channel_position[:, 0])] = -1
    
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

    def load_mua(self, file_idx=0, channel_idx=slice(0, 128), sample_range=None, scale=True):
        """
        Load mua data from binary file.

        Parameters
        ----------
        channel_idx : slice or list, optional
            The channels to load. If not provided, all channels are loaded.
        sample_range : tuple, optional
            The sample range to load. If not provided, all samples are loaded.
        scale : bool, optional
            Whether to scale the data to uV.

        Notes
        -----
        Check OneDrive/nclab/manual/ephys_intan/Intan_RHD2000_series_datasheet.pdf for more details.
        Amplifier Differential Gain: 192 V/V
        Amplifier AC Input Voltage Range: +/- 5 mV
        Theoretical Maximum Voltage (+/- 15 bits): 2**15 * 0.195 = +/- 6.390 mV
        Voltage Step Size of ADC (Least Significant Bit): 0.195 uV
        """
        tprint(f"Loading {self.mua_fn[file_idx]}")
        selected_channel = self.channel_id[channel_idx]
        data = read_bin(self.mua_fn[file_idx], n_channel=self.n_channel, dtype='int32', 
                        channel_idx=selected_channel, sample_range=sample_range)

        return data / (2 ** self.binary_radix) * self.uV_per_bit if scale else data
    
    def save_mua(self, channel_idx=slice(0, 128), right_shift=13, output_path=None):
        """
        Concatenate mua data to an int16 binary file.

        This method loads the MUA (Multi-Unit Activity) data from the original file(s),
        converts it to int16 format, and saves it to a single binary file. This is 
        typically done to prepare the data for spike sorting algorithms like Kilosort.

        Parameters:
        -----------
        channel_idx : slice or list, optional
            The channels to load. If not provided, the first 128 channels are loaded.
        output_path : str, optional
            The path where the concatenated int16 binary file will be saved.
            If not provided, it defaults to a 'kilosort' subdirectory in the current path.

        Notes:
        ------
        - The method assumes that the original data is in int32 format and needs to be 
          converted to int16.
        - The conversion process involves bit-shifting and scaling to preserve the 
          signal quality while reducing the file size.
        - uV/bit will be 1.56 for Intan RHD2000 series (if the right_shift is 16 bits and just used the upper 16 bits <- 0.195 uV/bit * 8).
        - Considering Neuropixels 2.0's uV/bit is 3.784 (12 bits, range -2048 to 2047, +/- 2**11), this scaling factor is reasonable.
        - Neuropixels 1.0: 0.6V / 500 gain / 512 bits = 2.34 uV/bit
        """
        if self.mua_fn is None:
            raise ValueError("No MUA files found")

        if output_path is None:
            output_path = os.path.join(self.path, 'kilosort')
            output_fn = os.path.join(output_path, f'{self.session_name}_tcat.imec.ap.bin')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        self.output_path = output_path
        self.output_fn = output_fn

        self.channel_id_saved = self.channel_id[channel_idx]
        self.n_channel_saved = len(self.channel_id_saved)
        self.channel_position_saved = self.channel_position[channel_idx, :]
        self.shank_saved = self.shank[channel_idx]

        if hasattr(self, 'mua_fn') and len(self.mua_fn) > 1:
            self.mua_fn = inquirer.checkbox(message="Select files to merge (files are ordered by time)", choices=self.mua_fn, default=self.mua_fn)
        
        if not self.mua_fn:
            return
        
        if len(self.mua_fn) > 1:
            self.mua_fn = file_reorder(self.mua_fn)

        self.save_catgt()
        
        if os.path.exists(self.output_fn):
            if inquirer.confirm("Redo saving?", default=False):
                os.remove(self.output_fn)
            else:
                print("The file is already saved. Exiting.")
                return

        with open(output_fn, 'wb') as f:
            for i, fn in enumerate(self.mua_fn):
                tprint(f"Processing {fn}")
                start_time = time.time()

                # 16.20 seconds preallocate (fastest but takes huge memory)
                # data = np.memmap(fn, dtype='int16', mode='r', shape=(self.n_sample[i], 2*self.n_channel))
                # output_buffer = np.empty((self.n_sample[i], len(self.channel_id_saved)), dtype=np.int16)
                # np.copyto(output_buffer, data[:, 2*self.channel_id_saved-1])
                # output_buffer.tofile(f)

                # 18.45 seconds preallocate (fastest but takes huge memory) 456.68 MB/s
                data = np.memmap(fn, dtype='int32', mode='r', shape=(self.n_sample[i], self.n_channel))
                output_buffer = np.empty((self.n_sample[i], len(self.channel_id_saved)), dtype=np.int16)
                np.right_shift(data[:, self.channel_id_saved], right_shift, out=output_buffer)
                output_buffer.tofile(f)

                # 174.76 seconds
                # with open(fn, 'rb') as source_file:
                #     chunk_size = 160 * 1024 * 1024  # 160 MB chunks (160M * 4 bytes for int32)
                #     while True:
                #         chunk = np.fromfile(source_file, dtype='int32', count=chunk_size)
                #         if chunk.size == 0:
                #             break
                #         chunk = chunk.reshape(-1, self.n_channel)[:, self.channel_id_saved]
                #         chunk = np.right_shift(chunk, 13).astype(np.int16)
                #         chunk.tofile(f)

                # 140.90 seconds
                # with open(fn, 'rb') as source_file:
                #     chunk_size = 160 * 1024 * 1024  # 160 MB chunks
                #     while True:
                #         chunk = np.fromfile(source_file, dtype='int32', count=chunk_size)
                #         if chunk.size == 0:
                #             break
                #         chunk.view(np.int16)[1::2].tofile(f)


                # 151.59 seconds 187.15 MB/s (super slow...)
                # data = np.memmap(fn, dtype='int16', mode='r')
                # data[1::2].tofile(f)

                # 133.88 seconds memmap
                # data = np.memmap(fn, dtype='int32', mode='r', shape=(self.n_sample[i], self.n_channel))
                # np.right_shift(data[:, self.channel_id_saved], 13).astype(np.int16).tofile(f)
                # del data  # Close the memmap

                end_time = time.time()
                duration = end_time - start_time
                tprint(f"Finished saving {fn} in {duration:.2f} seconds ({os.path.getsize(fn) / 1024**2 / duration:.2f} MB/s)")
        self.save_meta()

    def save_meta(self):
        """
        Save a metadata file.
        """
        fn = self.output_fn
        filesize = os.path.getsize(fn)
        n_sample = np.array(filesize) // 2 // self.n_channel_saved 
        filetime = datetime.fromtimestamp(os.path.getmtime(fn)).strftime("%Y-%m-%dT%H:%M:%S")
        filetime_orig = datetime.fromtimestamp(os.path.getmtime(self.mua_fn[0])).strftime("%Y-%m-%dT%H:%M:%S")
        filetimesecs = n_sample / self.sample_rate

        def calc_sha1(fn):
            tprint(f"Calculating SHA1 for {fn}")
            BUF_SIZE = 1024 * 1024  # Increase buffer size to 1MB for better throughput
            
            sha1_hash = hashlib.sha1()
            with open(fn, 'rb') as f:
                while chunk := f.read(BUF_SIZE):  # More concise loop using walrus operator
                    sha1_hash.update(chunk)
            return sha1_hash.hexdigest()

        metadata = {
            "acqApLfSy": "384,384,1",
            "appVersion": "20240129",
            "fileCreateTime": filetime,
            "fileCreateTime_original": filetime_orig,
            "fileName": self.output_fn,
            "fileSHA1": calc_sha1(fn),
            "fileSizeBytes": filesize,
            "fileTimeSecs": f"{filetimesecs:.3f}",
            "imAiRangeMax": "1.22683392",
            "imAiRangeMin": "-1.22683392",
            "imChan0apGain": "192",
            "imDatPrb_pn": "PRB_1_4_0480_1_C",
            "imDatPrb_sn": "20097921810",
            "imDatPrb_type": "0",
            "imMaxInt": str(2**15), # But SpikeGLX seems to ignore this
            "imSampRate": self.sample_rate,
            "nSavedChans": self.n_channel_saved,
            "snsApLfSy": f"{self.n_channel_saved},0,0",
            "snsSaveChanSubset": f"0:{self.n_channel_saved - 1}",
            "typeThis": "imec",
            "~imroTbl": self.get_imrotbl(),
            "~snsChanMap": self.get_snschanmap(),
            "~snsGeomMap": self.get_snsgeommap()
        }

        with open(os.path.splitext(fn)[0] + '.meta', 'w') as f:
            for key, value in metadata.items():
                f.write(f'{key}={value}\n')


    def get_imrotbl(self):
        imrotbl = "(0,384)"
        for channel in range(384):
            imrotbl += f"({channel} 0 0 192 80 1)"
        return imrotbl

    def get_snschanmap(self):
        snschanmap = f"(384,384,1)"
        for i in range(self.n_channel_saved):
            snschanmap += f"(AP{i};{i}:{i})"
        return snschanmap

    def get_snsgeommap(self):
        # Note: x position will be set to 0 for all channels
        # To recontruct the x position, add the shank index * shank_spacing
        # We set the shank spacing to 1000 um.
        snsgeommap = f"(PRB_1_4_0480_1_C,{np.unique(self.shank_saved).size},1000,70)"
        for i, (shank, position) in enumerate(zip(self.shank_saved, self.channel_position_saved)):
            snsgeommap += f"({shank}:0:{int(position[1])}:{int(not np.isnan(position[0]))})"
        return snsgeommap

    def save_catgt(self):
        """
        Save metadata to a csv file.

        This method collects information about the MUA binary files and saves it to a CSV file.
        The metadata includes file names, modification times, file sizes, number of samples,
        and start indices for each file.

        Notes:
        ------
        - This method assumes that `self.mua_fn` contains the list of MUA binary file paths.
        - The CSV file is saved in the same directory as the MUA files, with the name 'meta.csv'.
        """
        mtime = [datetime.fromtimestamp(os.path.getmtime(x)).strftime("%Y%m%d_%H%M%S") for x in self.mua_fn]
        file_size = [os.path.getsize(x) for x in self.mua_fn]
        self.n_sample = [int(os.path.getsize(x) / (4 * self.n_channel)) for x in self.mua_fn]
        self.mua_start = np.concatenate([[0], np.cumsum(self.n_sample)[:-1]])
        # new_filesize = np.array(file_size) / 2 / self.n_channel * 128

        self.catgt = pd.DataFrame({
            'filename': self.mua_fn,
            'mtime': mtime,
            'filesize': file_size,
            'n_sample': self.n_sample,
            'start': self.mua_start
        })
        fn = self.output_fn.replace('.bin', '.csv')
        self.catgt.to_csv(fn, index=False)
        tprint(f"Metadata saved to {fn}")
    
    def plot_mua(self, file_idx=0, channel_idx=slice(0, 128), sample_range=(5000, 10000)):
        data = self.load_mua(file_idx=file_idx, channel_idx=channel_idx, sample_range=sample_range)

        data -= np.median(data, axis=0, keepdims=True)
        data -= np.median(data, axis=1, keepdims=True)

        plt.figure(figsize=(12, 8))
        for i in range(data.shape[1]):
            plt.plot(data[:, i] + i * 100, color='k', linewidth=0.5)
        plt.xlabel('Samples')
        plt.ylabel('Channel')
        plt.title('Raw data')
        plt.show()

    def load_spk(self, file_idx=0):
        if self.spk_fn:
            tprint(f"Loading {self.spk_fn[file_idx]}")
            spk = np.fromfile(self.spk_fn[file_idx], dtype='<i4').reshape(-1, 2)
            self.spk_frame = spk[:, 0]
            self.spk_channel = spk[:, 1]  # Note: not in physical order
    
    def load_spk_wav(self, file_idx=0):
        if self.spk_wav_fn:
            tprint(f"Loading {self.spk_wav_fn[file_idx]}")
            spk_wav = np.fromfile(self.spk_wav_fn[file_idx], dtype=np.int32).reshape(-1, 20, 4)
            self.spk_peak_ch, self.spk_time, self.electrode_group = spk_wav[..., 0, 1], spk_wav[..., 0, 2], spk_wav[..., 0, 3]
            self.spk_wav = spk_wav[..., 1:, :] / (2**self.binary_radix) * self.uV_per_bit
    
    def load_fet(self, file_idx=0):
        if self.fet_fn:
            tprint(f"Loading {self.fet_fn[file_idx]}")
            scale_factor = float(2**self.binary_radix) / self.uV_per_bit

            file_size = os.stat(self.fet_fn[file_idx]).st_size
            if file_size // 4 % 8 == 0:
                n_col = 8
            elif file_size // 4 % 7 == 0:
                n_col = 7
            else:
                raise ValueError(f'Unsupported fet file size: {file_size}')

            fet = np.fromfile(self.fet_fn[file_idx], dtype=np.int32).reshape(-1, n_col)
            self.fet = pd.DataFrame({
                'frame': fet[:, 0].astype(np.int64),
                'group_id': fet[:, 1],
                'fet0': fet[:, 2].astype(np.float32) / scale_factor,
                'fet1': fet[:, 3].astype(np.float32) / scale_factor,
                'fet2': fet[:, 4].astype(np.float32) / scale_factor,
                'fet3': fet[:, 5].astype(np.float32) / scale_factor,
                'spike_id': fet[:, 6],
            })
            if n_col == 8:
                self.fet['spike_energy'] = fet[:, 7].astype(np.float32) / scale_factor

            # Check for any potential issues in the loaded data
            if self.fet['frame'].diff().min() < 0:
                tprint("\033[91mWarning: Time values are not monotonically increasing.\033[0m")
                in_idx = np.where(self.fet['frame'].diff() < 0)[0][0]
                self.fet = self.fet.iloc[:in_idx].reset_index(drop=True)

            if np.any(np.isnan(self.fet.values)):
                tprint("\033[91mWarning: NaN values detected in fet data.\033[0m")
            if np.any(np.isinf(self.fet.values)):
                tprint("\033[91mWarning: Infinite values detected in fet data.\033[0m")

            # TODO: If frame rolls over, add 2^32 to the frame
    
    def load_model(self, file_idx=0):
        if self.model_fn and len(self.model_fn) > file_idx:
            tprint(f"Loading {self.model_fn[file_idx]}")
            self.model = pd.read_pickle(self.model_fn[file_idx])

    def load_nidq(self, path=None):
        """
        Load nidq data from meta file.

        Parameters
        ----------
        path : str, optional
            Path to the nidq meta file, by default None

        Note
        ----
        - There can be multiple nidq files in the session folder.
        """
        if path is None:
            path = self.path

        self.nidq_fn = finder(path, pattern=r'\.nidq.meta$', ask=False)

        if not self.nidq_fn:
            tprint("No nidq file found")
            return

        n_fn = len(self.nidq_fn)
        self.nidq = [None] * n_fn
        self.time_sync_nidq = [None] * n_fn
        for i, fn in enumerate(self.nidq_fn):
            self.nidq[i] = read_digital(fn)

            self.time_sync_nidq[i] = self.nidq[i].query("chan==4 and type==1").time.values
            time_sync_nidq_off = self.nidq[i].query("chan==4 and type==0").time.values
            tprint(f"Found {len(self.time_sync_nidq[i])} sync pulses")

            pulse_duration = time_sync_nidq_off - self.time_sync_nidq[i]
            if pulse_duration.min() < 0.090:
                tprint(f"Found sync pulse shorter than 90 ms: {pulse_duration.min()}")
            
            sync_func = sync(self.time_sync_nidq[i], self.time_sync_fpga)
            self.nidq[i]['time_fpga'] = sync_func(self.nidq[i]['time'])
        
    def save_nidq(self, path=None):
        if path is None:
            path = self.path
        
        if not hasattr(self, 'nidq_fn') or not hasattr(self, 'nidq'):
            self.load_nidq()

        for i, fn in enumerate(self.nidq_fn):
            tprint(f"Saving {fn}")
            self.nidq[i].to_pickle(fn.replace('.nidq.meta', '.nidq.pd'))

    def load_tdms(self, path=None, sync_channel=1):
        """
        Load event data from tdms file.
        """
        if path is None:
            path = self.path
        
        self.tdms_fn = finder(path, pattern=r'\.tdms$', multiple=True)

        if not self.tdms_fn:
            tprint("No tdms file found")
            return

        n_fn = len(self.tdms_fn)
        self.tdms = [None] * n_fn
        self.time_sync_tdms = [None] * n_fn
        for i, fn in enumerate(self.tdms_fn):
            self.tdms[i] = read_tdms(fn)
            self.time_sync_tdms[i] = self.tdms[i].query(f"chan=={sync_channel} and type==1").time.values
            time_sync_tdms_off = self.tdms[i].query(f"chan=={sync_channel} and type==0").time.values

            pulse_duration = time_sync_tdms_off - self.time_sync_tdms[i]
            if pulse_duration.min() < 0.090:
                tprint(f"Found sync pulse shorter than 90 ms: {pulse_duration.min()}")
            
            sync_func = sync(self.time_sync_tdms[i], self.time_sync_fpga)
            self.tdms[i]['time_fpga'] = sync_func(self.tdms[i]['time'])

    def save_tdms(self, path=None):
        if path is None:
            path = self.path
        
        if not hasattr(self, 'tdms_fn') or not hasattr(self, 'tdms'):
            self.load_tdms()

        for i, fn in enumerate(self.tdms_fn):
            tprint(f"Saving {fn}")
            self.tdms[i].to_pickle(fn.replace('.tdms', '.tdms.pd'))

if __name__ == '__main__':
    bmi = BMI('C:\\SGL_DATA')
    # bmi.plot_prb()
    bmi.save_mua()
    # bmi.load_spk()
    # bmi.load_spk_wav()
    # bmi.load_fet()
    # bmi.load_nidq()
    # bmi.save_nidq()
    breakpoint()