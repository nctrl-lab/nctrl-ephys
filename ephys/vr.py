import os
import json
import numpy as np
import pandas as pd

from .utils import finder, savemat_safe


class VR():
    def __init__(self, path=None):
        if path is None:
            path = finder(folder=False, multiple=False, pattern=r'.json$')
        
        self.path = path
        self.load()

    def load(self, path=None):
        path = path or self.path

        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")

        with open(path) as f:
            data = json.load(f)

        self.vr_file_name = path
        self.vr_time = os.path.getmtime(path)
        
        info_dict = {
            "monitor_info": next(x for x in data if "refreshRateHz" in x),
            "task_info": next(x for x in data if "animalName" in x),
            "task_parameter": next(x for x in data if "logTreadmill" in x)
        }
        self.__dict__.update(info_dict)
        
        self.log_info = data[-1]

        vr_data = [x for x in data if "position" in x]
        self.vr = {col: [] for col in vr_data[0].keys()}
        for item in vr_data:
            for key, value in item.items():
                if key == 'position':
                    self.vr[key].append([value['x'], value['y'], value['z']])
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
        
    def save(self, path=None):
        if path is None:
            path = finder(folder=False, multiple=False, pattern=r'.mat$')
        
        if path is None:
            print("No path provided. Please provide a path to save the data.")
            return

        data = {
            'vr': self.vr,
            'trial': self.trial,
            'task_info': self.task_info,
            'task_parameter': self.task_parameter,
            'monitor_info': self.monitor_info
        }
        
        savemat_safe(path, data)

if __name__ == "__main__":
    path = finder(path='C:\SGL_DATA', pattern='.json$')
    vr = VR(path=path)
    vr.save()