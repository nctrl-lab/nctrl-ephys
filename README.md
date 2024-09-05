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
3. Run Kilosort (`ephys runks`)
4. Run phy to curate the Kilosort results and save the result (`ephys saveks`)
    - This will also extract the sync and event times data from SpikeGLX files.
    - The final output will be a `.mat` file including the spike, sync, and event times (from nidq file) data.
    - NIDQ time will be synced with the spike data.
5. Extract behavioral data from VR log file (`ephys savevr`)
    - This will extract 'vr', 'trial', 'task_info', 'task_parameter', and 'monitor_info'.
    - It will be merged with the previous `.mat` file.
    - The final file will represent one behavioral session.
6. Load the data using `ephys.spike.Spike` class.
```python
spike = Spike(path='path/to/data.mat')
spike.plot() # this will generate an interactive raster and PSTH figure to browse the data.
```

## Pipeline for BMI data
1. Record BMI data
2. Run `ephys savebmi` to merge the BMI data into a binary file to run Kilosort.
3. Run Kilosort (`ephys runks`)
4. Run phy to curate the Kilosort results and save the result (`ephys saveks`)
5. Extract event data from data logger (`ephys savetdms`)
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

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.