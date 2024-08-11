import os
import re
from typing import Optional

import inquirer


def finder(path: str, pattern: str = r'.*meta$') -> Optional[str]:
    """
    Explore files in a given path and return a user-selected file matching the pattern.

    Args:
        path (str): The directory path to explore.
        pattern (str, optional): Regex pattern to match filenames. Defaults to r'.*meta$'.

    Returns:
        Optional[str]: The selected file path, or None if no files are found.
    """
    files = [
        os.path.join(root, filename)
        for root, _, filenames in os.walk(path)
        for filename in filenames
        if re.match(pattern, filename)
    ]

    if files:
        return inquirer.list_input("Select a file", choices=files)
    return None