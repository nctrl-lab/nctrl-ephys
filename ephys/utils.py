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
           multiple: bool = False,
           folder: bool = False,
           ask: bool = True,
           exclude_pattern: Optional[str] = r'_g\d+$') -> Optional[Union[str, List[str]]]:
    """
    Explore files in a given path and return a user-selected file matching the pattern.

    Args:
        path (Optional[str]): The directory path to explore. Defaults to None.
        pattern (str, optional): Regex pattern to match filenames. Defaults to '.meta'.
        multiple (bool, optional): Whether to allow multiple files to be selected. Defaults to False.
        folder (bool, optional): Whether to select folders or files. Defaults to False.
        ask (bool, optional): Whether to use a dialog or not. Defaults to True.
        exclude_pattern (Optional[str], optional): Regex pattern to exclude filenames. Defaults to None.

    Returns:
        Optional[Union[str, List[str]]]: The selected file path(s), or None if no files are found.
    """
    if path is None:
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)
        path = filedialog.askdirectory(title=f"Select a Directory to search for {pattern}")
        root.destroy()

    files = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(path)
        for filename in filenames
        if re.search(pattern, filename) and '.phy' not in os.path.dirname(os.path.join(root, filename))
    ]

    if folder:
        files = [re.sub(exclude_pattern, '', os.path.dirname(file)) for file in files]
        files = list(set(files))
        files.sort()  # Sort the list of folders alphabetically

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
                cmd = f"Select {'folders' if folder else 'files'} (Space: select/unselect, Ctrl-A: select all, Ctrl-R: unselect all, Enter: confirm)"
                return inquirer.checkbox(cmd, choices=files, default=files)
            else:
                cmd = f"Select a {'folder' if folder else 'file'} (Enter: confirm)"
                return inquirer.list_input(cmd, choices=files)

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
        self.listbox.insert(new_index, self.listbox.get(index))
        self.listbox.delete(index if index < new_index else index + 1)
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
        
        sio.savemat(fn, data_old)
    except Exception as e:
        print(f"Error {'loading or ' if os.path.exists(fn) else ''}saving file {fn}: {str(e)}")
        sio.savemat(fn, data_old)


def sync(time_a, time_b):
    """
    Synchronize two time series by finding a linear relationship between them.

    Parameters:
    -----------
    time_a : array-like
        First time series.
    time_b : array-like
        Second time series to synchronize with the first.

    Returns:
    --------
    callable
        A function that converts times from the first series to the second.

    Notes:
    ------
    This function performs the following steps:
    1. Checks if the input time series have the same length.
    2. Performs a linear regression to check if the relationship is linear.
    3. Removes outliers that deviate more than 2 ms from the linear fit.
    4. If there are enough remaining points, returns an interpolation function.
    5. If not enough points remain, returns the original linear regression function.

    If the r-squared value of the linear regression is less than 0.98, the function
    considers the sync to have failed and returns the identity function.
    """
    from scipy.stats import linregress
    from scipy.interpolate import interp1d

    if len(time_a) != len(time_b):
        tprint(f"Sync failed: time_a and time_b have different lengths: {len(time_a)} != {len(time_b)}")
        n_sync = min(len(time_a), len(time_b))
        time_a = time_a[:n_sync]
        time_b = time_b[:n_sync]
    
    # Check if the syncs are in linear relationship
    slope, intercept, r_value, _, _ = linregress(time_a, time_b)
    r_squared = r_value**2
    if r_squared < 0.98:
        tprint(f"Sync failed: slope {slope:.6f}, intercept {intercept:.6f}, r-squared {r_squared:.6f}")
        return lambda x: x
    tprint(f"Sync OK: slope {slope:.6f}, intercept {intercept:.6f}, r-squared {r_squared:.6f}")

    # Check if the syncs have any outliers
    sync_diff = time_a * slope + intercept - time_b
    outlier = sync_diff >= 0.002  # 2 ms
    if outlier.sum() > 0:
        time_a = time_a[~outlier]
        time_b = time_b[~outlier]
        tprint(f"Removed {outlier.sum()} outliers")

    # Check if there are enough sync points after removing outliers
    if len(time_a) < 2:
        tprint("Not enough sync points after removing outliers. Using original linear regression.")
        return lambda x: x * slope + intercept

    # Return the function to convert nidq time to imec time
    return interp1d(time_a, time_b, kind='linear', fill_value="extrapolate")


if __name__ == "__main__":
    # print(finder(folder=True, multiple=True, pattern=r'.bin$'))
    confirm("Are you sure you want to delete all files in this folder?")