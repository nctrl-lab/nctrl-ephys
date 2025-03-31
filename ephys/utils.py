import os
import re
import datetime
from typing import Optional, Union, List

from IPython import get_ipython
import tkinter as tk
from tkinter import filedialog
import inquirer


def tprint(text):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{timestamp}] {text}')


def finder(path: Optional[str] = None,
           pattern: str = r'\.meta$',
           msg: str = None,
           multiple: bool = False,
           folder: bool = False,
           ask: bool = True,
           exclude_pattern: Optional[str] = r'_g\d+$') -> Optional[Union[str, List[str]]]:
    """
    Explore files in a given path and return a user-selected file matching the pattern.

    Args:
        path (Optional[str]): The directory path to explore. Defaults to None.
        pattern (str, optional): Regex pattern to match filenames. Defaults to '.meta'.
        msg (str, optional): Message to display in the dialog. Defaults to None.
        multiple (bool, optional): Whether to allow multiple files to be selected. Defaults to False.
        folder (bool, optional): Whether to select folders or files. Defaults to False.
        ask (bool, optional): Whether to use a dialog or not. Defaults to True.
        exclude_pattern (Optional[str], optional): Regex pattern to exclude filenames.

    Returns:
        Optional[Union[str, List[str]]]: The selected file path(s), or None if no files are found.
    """
    if path is None:
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        path = filedialog.askdirectory(title=f"Select a Directory to search for {pattern}").replace('/', os.sep)
        root.destroy()

    files = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(path)
        for filename in filenames
        if re.search(pattern, filename) and '.phy' not in os.path.dirname(os.path.join(root, filename))
    ]

    files.sort(key=lambda x: os.path.getmtime(x))  # Sort the list of folders by the time of the folder creation

    if folder:
        files = [re.sub(exclude_pattern, '', os.path.dirname(file)) for file in files]
        files = list(set(files))

    if not files:
        return None
    
    if not ask:
        return files

    try:
        # If running in an IPython environment, use a Tkinter dialog for file selection
        if get_ipython() and 'IPKernelApp' in get_ipython().config:
            root = tk.Tk()
            root.title("Select File(s)")
            root.call('wm', 'attributes', '.', '-topmost', True)
            listbox = tk.Listbox(root, selectmode=tk.MULTIPLE if multiple else tk.SINGLE)
            listbox.pack(expand=True, fill=tk.BOTH)
            root.geometry("800x600")

            for file in files:
                listbox.insert(tk.END, file)

            tk.Button(root, text="Select", command=root.quit).pack()
            root.mainloop()

            selected_indices = listbox.curselection()
            root.destroy()

            if not selected_indices:
                return None
            
            if multiple:
                return [files[i] for i in selected_indices]
            else:
                return files[selected_indices[0]]

        # If running in a command line environment, use inquirer for file selection
        else:
            if multiple:
                if msg is None:
                    msg = f"Select {'folders' if folder else 'files'}"
                msg += f" (Space: select/unselect, Ctrl-A: select all, Ctrl-R: unselect all, Enter: confirm)"
                return inquirer.checkbox(msg, choices=files, default=files)
            else:
                if msg is None:
                    msg = f"Select a {'folder' if folder else 'file'}"
                msg += f" (Enter: confirm)"
                return inquirer.list_input(msg, choices=files)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def confirm(message: str, default: bool = False) -> bool:
    if get_ipython() and 'IPKernelApp' in get_ipython().config:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        result = tk.messagebox.askyesno(message, message)
        root.destroy()
        return result
    else:
        return inquirer.confirm(message, default=default)


class FileReorderApp:
    def __init__(self, root, file_list):
        self.root = root
        self.root.title("Reorder Files")
        self.root.geometry("600x480")
        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        self.listbox.pack(fill=tk.BOTH, expand=True)
        
        self.listbox.insert(tk.END, *file_list)
        
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.X)
        
        buttons = [
            ("△", self.move_up, tk.LEFT),
            ("▽", self.move_down, tk.LEFT),
            ("Finish", self.save_order, tk.RIGHT)
        ]
        for text, command, side in buttons:
            tk.Button(button_frame, text=text, command=command).pack(side=side, padx=5, pady=5)
        
        self.output = None
    
    def move_item(self, direction):
        selected_indices = self.listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Warning", "No file selected")
            return
        
        index = selected_indices[0]
        if (direction == -1 and index == 0) or (direction == 1 and index == self.listbox.size() - 1):
            return
        
        new_index = index + direction
        item = self.listbox.get(index)
        self.listbox.delete(index)
        self.listbox.insert(new_index, item)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(new_index)
        self.listbox.see(new_index)

    move_up = lambda self: self.move_item(-1)
    move_down = lambda self: self.move_item(1)
    
    def save_order(self):
        self.output = tuple(self.listbox.get(0, tk.END))
        self.root.destroy()


def file_reorder(file_list):
    root = tk.Tk()
    app = FileReorderApp(root, file_list)
    root.mainloop()
    return app.output


def savemat_safe(fn, data):
    import scipy.io as sio

    if not isinstance(data, dict):
        raise ValueError("data must be a dictionary")

    try:
        if os.path.exists(fn):
            data_old = sio.loadmat(fn, simplify_cells=True)
            if not isinstance(data_old, dict):
                raise ValueError(f"File {fn} is not a dictionary")

            overlap = False
            for key in data.keys():
                if key in data_old:
                    overlap = True
                    break
            
            if overlap and not confirm(f"Data {key} already exists. Do you want to overwrite it?"):
                return
            
            # merge data
            data_old.update(data)
        else:
            data_old = data
        
        sio.savemat(fn, data_old, oned_as='column')
    except Exception as e:
        print(f"Error {'loading or ' if os.path.exists(fn) else ''}saving file {fn}: {str(e)}")
        if locals().get('data_old', None) is not None:
            sio.savemat(fn, data_old, oned_as='column')


def sync(time_a, time_b, threshold=0.010):
    """
    Synchronize two time series by finding a linear relationship between them.

    Parameters:
    -----------
    time_a : array-like
        First time series.
    time_b : array-like
        Second time series to synchronize with the first.
    threshold : float, optional
        Threshold for removing outliers. Defaults to 10 ms.

    Returns:
    --------
    callable
        A function that converts times from the first series to the second.

    Notes:
    ------
    This function performs the following steps:
    1. Finds the closest matching points between the two time series.
    2. Performs a linear regression to check if the relationship is linear.
    3. Removes outliers that deviate more than threshold from the linear fit.
    4. If there are enough remaining points, returns an interpolation function.
    5. If not enough points remain, returns the original linear regression function.

    If the r-squared value of the linear regression is less than 0.98, the function
    considers the sync to have failed and returns the identity function.
    """
    import numpy as np
    from scipy.stats import linregress
    from scipy.interpolate import interp1d

    time_a, time_b = np.array(time_a), np.array(time_b)
    time_a0, time_b0 = time_a[0], time_b[0]
    
    # Normalize times to start at 0
    time_a_sync, time_b_sync = time_a - time_a0, time_b - time_b0
    
    # Handle different lengths efficiently
    if len(time_a) != len(time_b):
        tprint(f"Time_a and time_b have different lengths: {len(time_a)} != {len(time_b)}")
        tprint("Matching the number of points in time_a and time_b by finding closest corresponding timestamps...")
        if len(time_a_sync) > len(time_b_sync):
            indices = np.argmin(np.abs(time_b_sync[:, None] - time_a_sync), axis=1)
            time_a_sync = time_a_sync[indices]
        else:
            indices = np.argmin(np.abs(time_a_sync[:, None] - time_b_sync), axis=1)
            time_b_sync = time_b_sync[indices]
    
    # Calculate linear regression
    slope, intercept, r_value, _, _ = linregress(time_a_sync, time_b_sync)
    r_squared = r_value**2
    
    if r_squared < 0.98:
        tprint(f"Sync \033[91m(failed)\033[0m: slope {slope:.6f}, intercept {intercept:.6f}, r-squared {r_squared:.6f}")
        return lambda x: x
    
    tprint(f"Sync \033[92mOK\033[0m: slope {slope:.6f}, intercept {intercept:.6f}, r-squared {r_squared:.6f}")

    # Check for outliers
    sync_diff = time_a_sync * slope + intercept - time_b_sync
    outlier = np.abs(sync_diff) >= threshold
    
    if np.any(outlier):
        outlier_count = np.sum(outlier)
        time_a_clean, time_b_clean = time_a_sync[~outlier], time_b_sync[~outlier]
        tprint(f"Removed \033[91m{outlier_count}\033[0m outliers")
        
        # Recalculate regression with cleaned data
        if len(time_a_clean) >= 2:
            slope, intercept, r_value, _, _ = linregress(time_a_clean, time_b_clean)
            r_squared = r_value**2
            tprint(f"After outlier removal: slope {slope:.6f}, intercept {intercept:.6f}, r-squared {r_squared:.6f}")
    else:
        time_a_clean, time_b_clean = time_a_sync, time_b_sync

    # If not enough points after cleaning, use simple linear transformation
    if len(time_a_clean) < 2:
        tprint("Not enough sync points after removing outliers. Using original linear regression.")
        return lambda x: (x - time_a0) * slope + time_b0 + intercept

    # Return interpolation function for time conversion
    return interp1d(time_a_clean + time_a0, time_b_clean + time_b0, kind='linear', fill_value="extrapolate")


def rollover_recovery(data, max_value=2**32):
    import numpy as np

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    diffs = np.diff(data, prepend=data[0])
    rollovers = np.where(np.abs(diffs) > max_value // 2, -np.sign(diffs), 0)
    return data + np.cumsum(rollovers) * max_value


def get_file(key, pattern="matlab.exe", name=None, initialdir=None, reset=False):
    import keyring
    import tkinter as tk
    from tkinter import filedialog

    fn = keyring.get_password("nctrl", key)
    if fn and os.path.exists(fn) and not reset:
        print(f"Using {name} executable: {fn}")
        return fn

    pattern_ext = "*." + pattern.split('.')[-1]

    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    print(f"\033[1;36mSelect the {name} executable\033[0m")
    fn = filedialog.askopenfilename(
        title=f"Select the {name} executable ({pattern_ext})",
        filetypes=[("Executable file", pattern_ext)],
        initialdir=initialdir
    )
    root.destroy()

    if fn and os.path.exists(fn) and os.path.basename(fn) == pattern:
        keyring.set_password("nctrl", key, fn)
        return fn
    else:
        print(f"Invalid selection. Please choose the {name} executable ({pattern}).")
        return None


def get_path(key, name=None, initialdir=None, reset=False):
    import keyring
    import tkinter as tk
    from tkinter.filedialog import askdirectory

    fn = keyring.get_password("nctrl", key)
    if fn and os.path.exists(fn) and not reset:
        print(f"Using {name} path: {fn}")
        return fn

    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    print(f"\033[1;36mSelect the {name} path\033[0m")
    fn = askdirectory(
        title=f"Select the {name} path",
        initialdir=initialdir
    )
    root.destroy()

    if fn and os.path.exists(fn):
        keyring.set_password("nctrl", key, fn)
        return fn
    else:
        print(f"Invalid selection. Please choose the {name} path.")
        return None


if __name__ == "__main__":
    # print(finder(folder=True, multiple=True, pattern=r'.bin$'))
    # print(file_reorder(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']))
    print(get_path("ks2", "Kilosort2"))