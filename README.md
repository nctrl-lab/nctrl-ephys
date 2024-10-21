# nctrl-ephys

## Overview
`nctrl-ephys` is a collection of tools for electrophysiology data analysis. It includes functionalities for processing and analyzing data from various sources, including SpikeGLX and Kilosort.

## Installation
To install the package, use the following command:
```bash
pip install git+https://github.com/nctrl-lab/nctrl-ephys.git
```

## Requirements
- Python 3.9 or higher
- [SpikeGLX](https://billkarsh.github.io/SpikeGLX/)
- [CatGT](https://billkarsh.github.io/SpikeGLX/#catgt)
- [Kilosort4](https://github.com/Mouseland/Kilosort)
- [phy](https://github.com/cortex-lab/phy)

## Pipelines for SpikeGLX data
1. Run SpikeGLX to record Neuropixels data
2. Run CatGT (`ephys catgt`) to concatenate and denoise the data
3. Generate the probe map by reading meta file (`ephys probe`)
    - This will generate a `PROBE_TYPE.mat` file, which can be useful for the Kilosort GUI.
    - It also plots the probe map and channel numbers.
4. Run Kilosort (`ephys runks`)
    - Don't forget to switch the conda environment (`conda activate kilosort`) if the Kilosort is installed in a different environment.
    - If you would like to save metrics (L-ratio, isolation distance, waveform similarity, and overall scores), run `ephye runks --metric`.
5. Run phy to curate the Kilosort results (`conda activate phy2`)
6. Save the results (`ephys saveks`)
    - This will also extract sync and event times data from SpikeGLX files.
    - The final output will be a `.mat` file that includes the spike, sync, and event times (from the NIDQ file).
    - NIDQ time will be synced with the spike data.
7. Extract behavioral data from the VR log file (`ephys task`)
    - This will extract 'vr', 'trial', 'task_info', 'task_parameter', and 'monitor_info'.
    - The extracted data will be merged with the previous `.mat` file.
    - The final file will represent one behavioral session.
8. Load the data using `ephys.spike.Spike` class.

```python
from ephys.spike import Spike
from ephys.utils import finder

path = finder(msg='Select the .mat file', pattern='.mat$')
spike = Spike(path)
spike
```

```python
path
    C:\SGL_DATA\abc0\abc0_20240101_M1_g0_imec0\kilosort4\abc0_20240101_M1_g0_imec0_data.mat
spike
    time: (12,)
    frame: (12,)
    firing_rate: (12,)
    position: (12, 2)
    waveform: (12, 61, 14)
    waveform_idx: (12, 14)
    waveform_channel: (12, 14)
    waveform_position: (12, 14, 2)
    Vpp: (12,)
    n_unit: 12
    channel_map: (374,)
    channel_position: (374, 2)
    cluster_group: (370,)
    meta:
    n_channel: 384
    sample_rate: 29999.872727272726
    waveform_raw: (12, 61, 14)
    Vpp_raw: (12,)
sync
    time_imec: (407,)
    frame_imec: (407,)
    type_imec: (407,)
    time_nidq: (407,)
    frame_nidq: (407,)
    type_nidq: (407,)
nidq
    time: (408,)
    frame: (408,)
    chan: (408,)
    type: (408,)
    time_imec: (408,)
vr
    timeSecs: (29738,)
    frame: (29738,)
    timeSecsAfterSplash: (29738,)
    frameAfterSplash: (29738,)
    readTimestampMs: (29738,)
    speed: (29738,)
    rotation: (29738,)
    ballSpeed: (29738,)
    pitch: (29738,)
    roll: (29738,)
    yaw: (29738,)
    distance: (29738,)
    events: (29738,)
    position_x: (29738,)
    position_y: (29738,)
    position_z: (29738,)
trial
    timeSecs: (307,)
    frame: (307,)
    timeSecsAfterSplash: (307,)
    frameAfterSplash: (307,)
    iState: (307,)
    iTrial: (307,)
    iTrial1: (307,)
    iTrial2: (307,)
    iCorrect: (307,)
    iCorrect1: (307,)
    iCorrect2: (307,)
    iChoice: (307,)
    cChoice: (307,)
    iReward: (307,)
    delayDuration: (307,)
    rewardLatency: (307,)
    punishmentLatency: (307,)
    note: (307,)
......
```

```python
spike.plot() # this will generate an interactive raster and PSTH figure to browse the data.
```

## Pipeline for BMI data
1. Record BMI data
2. Run `ephys bmi` to merge the BMI data into a binary file to run Kilosort.
3. Run Kilosort (`ephys runks`)
4. Run phy to curate the Kilosort results and save the result (`ephys saveks --bmi`)
6. Load the data using `ephys.spike.Spike` class.


## Usage
### Command Line Interface
The package provides a command-line interface (CLI) for various operations. Below are some examples:

#### Running CatGT
```bash
ephys catgt --path /path/to/data
```
- You can omit the `--path` option and the command will ask you the path to the data.

#### Running Kilosort
```bash
ephys runks --path /path/to/data
```
- You can omit the `--path` option and the command will ask you the path to the data.

#### Saving Kilosort Results
- This command will generate a '.mat' file containing only the **good** units that were curated by phy.
- This command also saves the waveform data by reading the raw '.bin' files (by default, it will read the first 60 seconds of data).

```bash
ephys saveks --path /path/to/data
```

### Python API
You can also use the functionalities provided by `nctrl-ephys` directly in your Python scripts. Check the [example.ipynb](notebooks/example.ipynb) for more details.

#### Reading and plotting SpikeGLX Data
```python
from ephys.spikeglx import read_meta, read_analog, read_digital
from ephys.utils import finder

# Finding the data file
fn = finder("C:\\SGL_DATA")

# Loading the meta data
meta = read_meta(fn)

# Loading the Neuropixels data
data = read_analog(fn, sample_range=(0, 3000))

# Plotting the Neuropixels data
plt.imshow(data.T, vmin=-200, vmax=200, cmap='bwr', aspect='auto', interpolation='none')
plt.colorbar()
plt.title('Raw Data')
plt.xlabel('Time (samples)')
plt.ylabel('Channel')
plt.show()

# Loading digital data to get the sync pulse times
data_event = read_digital(fn)
time_sync = data_event.query('chan == 6').times.values
```

#### Running Kilosort
```python
from ephys.ks import run_ks4
run_ks4(path='/path/to/data')
```

#### Loading and Plotting MUA Data
```python
from ephys.bmi import BMI

bmi = BMI(path='/path/to/data')
bmi.load_mua()
bmi.plot_mua()
```

#### Ploting raster and PSTH
```python
# load mat file
path = finder(path="C:\SGL_DATA", msg='Select a session file', pattern=r'.mat$')
spike = Spike(path)

# plot raster and psth
time_spike = spike.spike['time'][0]
time_event = spike.nidq.query('chan == 2 and type == 1')['time_imec'].values
plot_raster_psth(time_spike, time_event)
plt.show()
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
