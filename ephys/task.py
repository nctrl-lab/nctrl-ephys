import os
import time
import json
import numpy as np

from .spike import smooth
from .utils import finder, savemat_safe, rollover_recovery


class Task():
    def __init__(self, path=None, task_type='unity'):
        if path is None:
            if task_type == 'unity':
                path = finder(folder=False, multiple=False, pattern=r'.json$')
            elif task_type == 'pi':
                path = finder(folder=False, multiple=False, pattern=r'.txt$')
        
        self.task_path = path
        self.task_type = task_type
        self.load(path, task_type)

    def load(self, path=None, task_type=None):
        path = path or self.task_path
        task_type = task_type or self.task_type

        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        
        if task_type is None:
            raise ValueError("Task type is required")

        if task_type == 'unity':
            self.parse_unity(path)
        elif task_type == 'pi':
            self.parse_pi(path)
    
    def parse_unity(self, path):
        with open(path) as f:
            data = json.load(f)

        self.task_time = os.path.getmtime(path)
        
        info_dict = {
            "monitor_info": next(x for x in data if "refreshRateHz" in x),
            "task_info": next(x for x in data if "animalName" in x),
            "task_parameter": next(x for x in data if "logTreadmill" in x)
        }
        self.__dict__.update(info_dict)
        
        self.log_info = data[-1]

        vr_data = [x for x in data if "position" in x]
        self.vr = {key: [] for key in vr_data[0].keys() if key != 'position'}
        self.vr['position_x'] = []
        self.vr['position_y'] = []
        self.vr['position_z'] = []
        
        for item in vr_data:
            for key, value in item.items():
                if key == 'position':
                    self.vr['position_x'].append(value['x'])
                    self.vr['position_y'].append(value['y'])
                    self.vr['position_z'].append(value['z'])
                elif key == 'events':
                    self.vr[key].append(value if value else '')
                else:
                    self.vr[key].append(value)
        
        for key, value in self.vr.items():
            if key == 'events':
                self.vr[key] = np.array(value, dtype=object)
            elif key == 'position':
                self.vr[key] = np.array(value)
            else:
                self.vr[key] = np.array(value, dtype=np.float64)

        # Optimize trial data processing
        trial_data = [x for x in data if "iState" in x]
        self.trial = {col: [] for col in trial_data[0].keys()}
        for item in trial_data:
            for key, value in item.items():
                self.trial[key].append(value)
        
        # Convert trial data to numpy arrays efficiently
        for key, value in self.trial.items():
            self.trial[key] = np.array(value)

    def parse_pi(self, path):
        self.task_time = os.path.getmtime(path)

        with open(path) as f:
            data = f.readlines()

        # Use list comprehension for faster processing
        parsed_data = [(int(c), int(t), list(map(int, vs)))
                       for line in data
                       if not line.startswith("0,")
                       for c, t, *vs in [line.strip().split(",")]]

        cmds, times, values = zip(*parsed_data)
        times = rollover_recovery(times) / 1e6 # seconds

        # Use defaultdict for more efficient data collection
        from collections import defaultdict
        data_types = defaultdict(list)

        for cmd, t, vs in zip(cmds, times, values):
            if 30 <= cmd < 40:
                data_types['vr'].append([t] + vs)
            elif 40 <= cmd < 50:
                data_types['sync'].append([cmd-40, t] + vs)
            elif 60 <= cmd < 70:
                data_types['trial'].append([cmd-60, t] + vs)
            elif 70 <= cmd < 80:
                data_types['laser'].append([cmd-70, t])
            elif 80 <= cmd < 90:
                data_types['reward'].append(t)

        N_PULSE_PER_CM = 8.1487
        vr_data, sync_data, trial_data, laser_data, reward_data = map(np.array, (data_types['vr'], data_types['sync'], data_types['trial'], data_types['laser'], data_types['reward']))

        if len(vr_data) > 0:
            self.vr = {
                "time": vr_data[:, 0],
                "position_raw": rollover_recovery(vr_data[:, 1])
            }
            self.vr['position'] = self.vr['position_raw'] / N_PULSE_PER_CM
            self.vr['speed'] = np.ediff1d(self.vr['position']) / np.ediff1d(self.vr['time']) # cm/s
            self.vr['speed_conv'] = smooth(self.vr['speed'], axis=0)

        if len(sync_data) > 0:
            self.trial_sync = {
                "time_task": sync_data[:, 1],
                "type_task": sync_data[:, 0]
            }

        if len(trial_data) > 0:
            self.trial = {
                "time": trial_data[1:, 1],
                "state": trial_data[1:, 2],
                "i_trial": trial_data[1:, 3]
            }
            # Successful trial: 1 -> 3
            # Failed trial: 1 -> 2 -> 4
            self.trial['timeStart'] = self.trial['time'][self.trial['state'] == 1]
            self.trial['timeEnd'] = self.trial['time'][self.trial['state'] == 2]
            self.trial['timeITISuccess'] = self.trial['time'][self.trial['state'] == 3]
            self.trial['timeITIFail'] = self.trial['time'][self.trial['state'] == 4]
            self.trial['result'] = 4 - self.trial['state'][(self.trial['state'] == 3) | (self.trial['state'] == 4)]
            self.trial['n_trial'] = len(self.trial['result'])

        if len(laser_data) > 0:
            self.laser = {
                "time": laser_data[:, 1],
                "type": laser_data[:, 0]
            }

        if len(reward_data) > 0:
            self.reward = {
                "time": reward_data
            }
        
    def save(self, path=None):
        if path is None:
            path = finder(folder=False, multiple=False, pattern=r'.mat$')
            if path is None:
                print("No path provided. Saving .mat file in the current directory.")
                path = self.task_path.replace('.txt', '.mat')

        data = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}
        savemat_safe(path, data)
    
    def summary(self):
        print(f"Task type: {self.task_type}")
        print(f"Task path: {self.task_path}")
        print(f"Task time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.task_time))}")

        if self.task_type == 'pi':
            print(f"Task duration: {self.vr['time'][-1] - self.vr['time'][0]:.2f} s")
            print(f"VR data: {len(self.vr['time'])}")
            print(f"Sync data: {len(self.trial_sync['time_task'])}")
            print(f"Trial data:")
            print(f"    n_trial: {self.trial['n_trial']}")
            print(f"    performance: {np.mean(self.trial['result'] == 1) * 100:.2f}%")
            print(f"    reward: {len(self.reward['time']) * 0.02:.2f} mL (n={len(self.reward['time'])})")

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        color = ['lightgray', 'gray']
        for i in range(2):
            in_trial = np.where(self.trial['result'] == i)[0]
            plt.bar(in_trial + 0.5, i * np.ones_like(in_trial) * 0.8 + 0.2, width=1, color=color[i])
        result_conv = [0] + list(smooth(self.trial['result'], axis=0, mode='full', type='boxcar') / 10)
        plt.step(np.arange(len(result_conv)), result_conv, color='r')
        plt.xlim(0, self.trial['n_trial'])
        plt.ylim(0, 1)
        plt.xlabel('Trial number')
        plt.ylabel('Performance')
        plt.show()

if __name__ == "__main__":
    path = finder(path='C:\SGL_DATA', pattern='.txt$')
    task = Task(path=path, task_type='pi')
    task.plot()
    # task.save()