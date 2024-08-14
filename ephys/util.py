import os
import re
from typing import Optional

from IPython import get_ipython
import tkinter as tk
from tkinter import filedialog
import inquirer


def finder(path: Optional[str] = None, pattern: str = r'\.meta$') -> Optional[str]:
    """
    Explore files in a given path and return a user-selected file matching the pattern.

    Args:
        path (Optional[str]): The directory path to explore. Defaults to None.
        pattern (str, optional): Regex pattern to match filenames. Defaults to r'\.meta'.

    Returns:
        Optional[str]: The selected file path, or None if no files are found.
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
        if re.match(r'.*' + pattern, filename)
    ]

    if files:
        try:
            # If running in an IPython environment, use a Tkinter dialog for file selection
            if get_ipython() and 'IPKernelApp' in get_ipython().config:
                root = tk.Tk()
                root.title("Select a File")
                root.call('wm', 'attributes', '.', '-topmost', True)
                listbox = tk.Listbox(root, selectmode=tk.SINGLE)
                listbox.pack(expand=True, fill=tk.BOTH)
                root.geometry("800x600")

                for file in files:
                    listbox.insert(tk.END, file)

                tk.Button(root, text="Select File", command=root.quit).pack()
                root.mainloop()

                try:
                    selected_index = listbox.curselection()
                    root.destroy()
                    return files[selected_index[0]] if selected_index else None
                except:
                    return None
            # If running in a command line environment, use inquirer for file selection
            else:
                return inquirer.list_input("Select a file", choices=files)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    return None


if __name__ == "__main__":
    print(finder())