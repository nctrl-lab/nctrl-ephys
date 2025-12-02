import os
import time
import json
import numpy as np

from .spike import smooth
from .utils import finder, savemat_safe, rollover_recovery, file_reorder


class Task:
    def __init__(self, path=None, task_type='unity'):
        if path is None:
            pattern = r'.json$' if task_type == 'unity' else r'.txt$'
            path = finder(folder=False, multiple=True, pattern=pattern)
            if path and len(path) > 1:
                path = file_reorder(path)
        self.task_path = path
        self.n_file = len(path)
        self.task_type = task_type
        self.load(path, task_type)

    def load(self, path=None, task_type=None):
        path = path or self.task_path
        task_type = task_type or self.task_type

        parsers = {'unity': self.parse_unity, 'pi': self.parse_pi}
        if task_type not in parsers:
            raise ValueError(f"Unsupported task type: {task_type}")

        for file_path in path:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist")
            parsers[task_type](file_path)
        

    def parse_unity(self, path):
        with open(path) as f:
            data = json.load(f)
        task_time = os.path.getmtime(path)

        def get_first(key):
            for x in data:
                if key in x: return x
            raise KeyError(f"Field {key} not found in data.")

        info_dict = {
            "monitor_info": get_first("refreshRateHz"),
            "task_info": get_first("animalName"),
            "task_parameter": get_first("logTreadmill"),
        }
        log_info = data[-1]

        vr_data = [x for x in data if "position" in x]
        vkeys = [k for k in vr_data[0] if k != 'position']
        vr_dict = {key: [] for key in vkeys}
        vr_dict.update({'position_x': [], 'position_y': [], 'position_z': []})

        for item in vr_data:
            pos = item.get('position', None)
            if pos is not None:
                vr_dict['position_x'].append(pos.get('x', np.nan))
                vr_dict['position_y'].append(pos.get('y', np.nan))
                vr_dict['position_z'].append(pos.get('z', np.nan))
            for key, value in item.items():
                if key == "events":
                    vr_dict[key].append(value if value else '')
                elif key != 'position':
                    vr_dict[key].append(value)

        for key in vr_dict:
            arr_type = object if key == 'events' else np.float64
            vr_dict[key] = np.array(vr_dict[key], dtype=arr_type)

        trial_data = [x for x in data if "iState" in x]
        trial_keys = trial_data[0].keys()
        trial_dict = {k: np.array([item[k] for item in trial_data]) for k in trial_keys}

        # Only keep up to and including the last result trial
        in_result = (trial_dict['iState'] == 4) | (trial_dict['iState'] == 5)
        i_end = np.where(in_result)[0][-1]
        trial_dict = {k: v[:i_end+1] for k,v in trial_dict.items()}

        iState = trial_dict['iState']
        trial_dict['timeStartVr']   = trial_dict['timeSecs'][iState == 2]
        trial_dict['timeChoiceVr']  = trial_dict['timeSecs'][iState == 3]
        trial_dict['timeSuccessVr'] = trial_dict['timeSecs'][iState == 4]
        trial_dict['timeFailVr']    = trial_dict['timeSecs'][iState == 5]

        result_mask = (iState == 4) | (iState == 5)
        trial_dict['timeResultVr'] = trial_dict['timeSecs'][result_mask]
        trial_dict['cue']    = trial_dict['cChoice'][result_mask]
        trial_dict['choice'] = trial_dict['iChoice'][result_mask]
        trial_dict['result'] = 5 - trial_dict['iState'][result_mask]
        trial_dict['reward'] = trial_dict['iReward'][result_mask]
        
        n_trial = min(len(trial_dict['timeStartVr']), len(trial_dict['timeChoiceVr']), len(trial_dict['timeResultVr']))
        trial_dict = {k: v[:n_trial] for k,v in trial_dict.items()}
        trial_dict['n_trial'] = n_trial

        multi_save = (getattr(self, 'n_file', 1) > 1)
        def save_attr(attr, val):
            # assign as list if multi; append or make new list as needed
            if multi_save:
                if not hasattr(self, attr) or not isinstance(getattr(self, attr), list):
                    setattr(self, attr, [])
                getattr(self, attr).append(val)
            else:
                setattr(self, attr, val)

        save_attr('task_time', task_time)
        save_attr('monitor_info', info_dict["monitor_info"])
        save_attr('task_info', info_dict["task_info"])
        save_attr('task_parameter', info_dict["task_parameter"])
        save_attr('log_info', log_info)

        trial_dict['timeBinned'] = self.space_binning_for(trial_dict, vr_dict)
        save_attr('vr', vr_dict)
        save_attr('trial', trial_dict)

    def space_binning_for(self, trial, vr):
        task_info = self.task_info
        if isinstance(task_info, list):
            if len(task_info) == 0:
                raise ValueError("task_info list is empty")
            task = task_info[-1]['task']
        else:
            task = task_info['task']

        if task == 'Beacon':
            delay_bin = np.arange(310, 580, 10, dtype=float)
            choice_bin = np.arange(610, 1330, 10, dtype=float)
        elif task == 'Alter':
            delay_bin = np.arange(-240, 180, 10, dtype=float)
            choice_bin = np.arange(210, 700, 10, dtype=float)
        else:
            raise ValueError("Unknown task type for binning.")

        t = vr['timeSecs']
        z = vr['position_z']
        n_trial = trial['n_trial']

        idx_start  = np.searchsorted(t, trial['timeStartVr'])
        idx_choice = np.searchsorted(t, trial['timeChoiceVr'])
        idx_result = np.searchsorted(t, trial['timeResultVr'])

        time_bin_delay = np.full((n_trial, len(delay_bin)), np.nan)
        time_bin_choice = np.full((n_trial, len(choice_bin)), np.nan)

        for i in range(n_trial):
            s0, s1, s2 = idx_start[i], idx_choice[i], idx_result[i]
            time_bin_delay[i] = find_first_crossing(z[s0:s1], t[s0:s1], delay_bin)
            time_bin_choice[i] = find_first_crossing(z[s1:s2], t[s1:s2], choice_bin)

        return np.concatenate(
            (trial['timeStartVr'][:, None], time_bin_delay, trial['timeChoiceVr'][:, None],
             time_bin_choice, trial['timeResultVr'][:, None]), axis=1)

    def parse_pi(self, path):
        from collections import defaultdict
        task_time = os.path.getmtime(path)
        with open(path) as f:
            data = [line.strip() for line in f if not line.startswith("0,")]

        # Parse all lines efficiently
        cmds, times, values = [], [], []
        for line in data:
            toks = line.split(",")
            cmds.append(int(toks[0]))
            times.append(int(toks[1]))
            values.append(list(map(int, toks[2:])))
        times = rollover_recovery(times) / 1e6  # seconds

        data_types = defaultdict(list)
        for cmd, t, vs in zip(cmds, times, values):
            if 30 <= cmd < 40:
                data_types['vr'].append([t] + vs)
            elif 40 <= cmd < 50:
                data_types['sync'].append([cmd - 40, t] + vs)
            elif 60 <= cmd < 70:
                data_types['trial'].append([cmd - 60, t] + vs)
            elif 70 <= cmd < 80:
                data_types['laser'].append([cmd - 70, t])
            elif 80 <= cmd < 90:
                data_types['reward'].append(t)
        to_array = lambda x: np.array(x) if len(x) else np.empty((0,))

        N_PULSE_PER_CM = 8.1487
        vr_data, sync_data, trial_data, laser_data, reward_data = map(
            to_array, [data_types['vr'], data_types['sync'],
                       data_types['trial'], data_types['laser'],
                       data_types['reward']]
        )

        multi_save = (getattr(self, 'n_file', 1) > 1)
        def save_attr(attr, val):
            if multi_save:
                if not hasattr(self, attr) or not isinstance(getattr(self, attr), list):
                    setattr(self, attr, [])
                getattr(self, attr).append(val)
            else:
                setattr(self, attr, val)

        vr_dict = sync_dict = trial_dict = laser_dict = reward_dict = None
        if vr_data.size:
            pos_raw = rollover_recovery(vr_data[:,1])
            time_vec = vr_data[:,0]
            position = pos_raw / N_PULSE_PER_CM
            dtime = np.diff(time_vec)
            ds = np.diff(position)
            speed = np.concatenate(([0], ds / dtime))
            vr_dict = dict(
                time=time_vec,
                position_raw=pos_raw,
                position=position,
                speed=speed,
                speed_conv=smooth(speed, axis=0, sigma=5)
            )
        if sync_data.size:
            sync_dict = dict(time_task=sync_data[:,1], type_task=sync_data[:,0])
        if trial_data.size:
            td = trial_data
            trial_dict = dict(
                time=td[1:,1],
                state=td[1:,2],
                i_trial=td[1:,3]
            )
            s = trial_dict['state']
            trial_dict['timeStartVr'] = trial_dict['time'][s == 1]
            trial_dict['timeEndVr'] = trial_dict['time'][s == 2]
            trial_dict['timeITISuccessVr'] = trial_dict['time'][s == 3]
            trial_dict['timeITIFailVr'] = trial_dict['time'][s == 4]
            in_result = (s == 3) | (s == 4)
            trial_dict['result'] = 4 - s[in_result]
            trial_dict['n_trial'] = len(trial_dict['result'])
        if laser_data.size:
            laser_dict = dict(time=laser_data[:,1], type=laser_data[:,0])
        if reward_data.size:
            reward_dict = dict(time=reward_data)

        save_attr('task_time', task_time)
        save_attr('vr', vr_dict)
        save_attr('trial_sync', sync_dict)
        save_attr('trial', trial_dict)
        save_attr('laser', laser_dict)
        save_attr('reward', reward_dict)

    def save(self, path=None):
        if path is None:
            path = finder(folder=False, multiple=False, pattern=r'.mat$')
            if path is None and hasattr(self, "task_path"):
                print("No path provided. Saving .mat file in the current directory.")
                path = self.task_path.replace('.txt', '.mat')
        # Only export non-private attributes
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        savemat_safe(path, data)

    def summary(self):
        print(f"Task type: {self.task_type}")
        print(f"n_file: {getattr(self, 'n_file', 1)}")
        print(f"Task path: {self.task_path}")

        for i in range(self.n_file):
            multi = self.n_file > 1
            vr = self.vr[i] if multi else self.vr
            trial = self.trial[i] if multi else self.trial
            
            if self.task_type == 'pi':
                trial_sync = self.trial_sync[i] if multi else self.trial_sync
                reward = self.reward[i] if multi else self.reward
                print(f"\nBlock {i+1}:")
                print(f"Task duration: {vr['time'][-1] - vr['time'][0]:.2f} s")
                print(f"VR data: {len(vr['time'])}")
                print(f"Sync data: {len(trial_sync['time_task'])}")
                print("Trial data:")
                print(f"    n_trial: {trial['n_trial']}")
                perf = np.mean(trial['result'] == 1) * 100 if trial['n_trial'] else 0
                print(f"    performance: {perf:.2f}%")
                n_reward = len(reward['time'])
                print(f"    reward: {n_reward * 0.02:.2f} mL (n={n_reward})")
            elif self.task_type == 'unity':
                print(f"\nBlock {i+1}:")
                print(f"Task duration: {vr['timeSecs'][-1] - vr['timeSecs'][0]:.2f} s")
                print(f"VR data: {len(vr['timeSecs'])}")
                print("Trial data:")
                print(f"    n_trial: {trial['n_trial']}")
                perf = np.mean(trial['result'] == 1) * 100 if trial['n_trial'] else 0
                print(f"    performance: {perf:.2f}%")
                print(f"    reward: {trial['reward'][-1] * 0.001:.2f} mL")

    def plot(self):
        import matplotlib.pyplot as plt

        multi = getattr(self, 'n_file', 1) > 1
        n_blocks = self.n_file if hasattr(self, 'n_file') else (len(self.trial) if isinstance(self.trial, list) else 1)

        if multi or (isinstance(self.trial, list) and len(self.trial) > 1):
            # Concatenate results from all blocks
            all_results = []
            block_lengths = []
            for trial in self.trial:
                result = np.asarray(trial['result'])
                all_results.append(result)
                block_lengths.append(len(result))
            result_cat = np.concatenate(all_results)
            n_trial = result_cat.size
            # Block borders are at the trial cumulative lengths (excluding 0 and n_trial)
            block_edges = np.cumsum([0] + block_lengths)
            # Avoid first 0, last n_trial (already used for xlim)
            block_borders = block_edges[1:-1]
        else:
            result_cat = np.asarray(self.trial['result'])
            n_trial = result_cat.size
            block_borders = []

        plt.figure()
        colors = ['lightgray', 'gray']

        # Plot correct/incorrect bars
        for i, color in enumerate(colors):
            in_trial = np.where(result_cat == i)[0]
            if in_trial.size > 0:
                plt.bar(in_trial + 0.5, i * 0.8 + 0.2, width=1, color=color, edgecolor='k', linewidth=0.5)

        # Plot vertical lines at block borders if there are multiple blocks
        for ix in block_borders:
            plt.axvline(ix, color='b', linestyle='--', linewidth=1, alpha=0.6, zorder=2)

        # Compute smoothed performance (boxcar/rolling mean over 10 trials)
        trial_perf = smooth(result_cat, axis=0, mode='full', type='boxcar') / 10
        result_conv = np.concatenate(([0], trial_perf))
        plt.step(np.arange(len(result_conv)), result_conv, color='r', where='mid')

        plt.xlim(0, n_trial)
        plt.ylim(-0.05, 1.05)
        plt.xlabel('Trial number')
        plt.ylabel('Performance')
        if multi or (isinstance(self.trial, list) and len(self.trial) > 1):
            plt.title(f'Task Performance Over Trials (Blocks: {n_blocks})')
        else:
            plt.title('Task Performance Over Trials')
        plt.tight_layout()
        plt.show()

def find_first_crossing(x, t, x_bins):
    """
    Find the first crossing of a line with a given slope.

    Parameters
    ----------
    x : array-like
        The position coordinates of the line.
    t : array-like
        The time coordinates of the line.
    x_bins : array-like
        The position bins to find the first crossing of.

    Returns
    -------
    result : array-like
        The time of the first crossing of the line with the given position bins.
    """
    x_diff = np.diff(x)
    t_diff = np.diff(t)
    x_prev = x[:-1]
    t_prev = t[:-1]

    result = np.full_like(x_bins, np.nan)
    for j, b in enumerate(x_bins):
        crossings = (x_prev < b) & (x_prev + x_diff >= b) & (x_diff > 0)
        if np.any(crossings):
            idx = np.argmax(crossings)
            t_frac = (b - x_prev[idx]) / x_diff[idx]
            result[j] = t_prev[idx] + t_frac * t_diff[idx]
    return result

if __name__ == "__main__":
    task = Task()
    task.summary()
    # task.plot()
    task.save()