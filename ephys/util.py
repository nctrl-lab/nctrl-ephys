import os
import re
from typing import Optional

from IPython import get_ipython
import tkinter as tk
from tkinter import filedialog
import inquirer


def finder(path: Optional[str] = None,
           pattern: str = r'\.meta$',
           multiple: bool = False,
           folder: bool = False) -> Optional[str | list[str]]:
    """
    Explore files in a given path and return a user-selected file matching the pattern.

    Args:
        path (Optional[str]): The directory path to explore. Defaults to None.
        pattern (str, optional): Regex pattern to match filenames. Defaults to '.meta'.
        multiple (bool, optional): Whether to allow multiple files to be selected. Defaults to False.
        folder (bool, optional): Whether to select folders or files. Defaults to False.

    Returns:
        Optional[str | list[str]]: The selected file path(s), or None if no files are found.
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
        if re.search(pattern, filename)
    ]

    if folder:
        files = [os.path.dirname(file) for file in files]
        files = list(set(files))
        files.sort()  # Sort the list of folders alphabetically

    if not files:
        return None

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
                return inquirer.checkbox("Select folders" if folder else "Select files (Space: select/unselect, Ctrl-A: select all, Ctrl-R: unselect all, Enter: confirm)", choices=files, default=files)
            else:
                return inquirer.list_input("Select a folder" if folder else "Select a file (Space: select/unselect, Enter: confirm)", choices=files)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
if __name__ == "__main__":
    print(finder(folder=True, multiple=True, pattern=r'\.nidq\.meta$'))