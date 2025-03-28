"""CatGT processing module.

This module provides functionality for working with CatGT, a tool for processing
electrophysiology data. It includes functions for selecting the CatGT executable,
managing file paths, and processing session data.

By default, we ignore real-world time gaps (zerofillmax=0).

SpikeGLX records files in the following order:
    g0_t0, g0_t1, g0_t2, ...
    g1_t0, g1_t1, g1_t2, ...
    g2_t0, g2_t1, g2_t2, ...

Download CatGT from https://billkarsh.github.io/SpikeGLX/
"""

import re
import glob
import os
import inquirer
import subprocess

from .utils import finder, get_file

def get_sessions(path=None):
    """Find and return sessions that can be processed by CatGT.
    
    Args:
        path (str, optional): Directory path to search. If None, will prompt for selection.
        
    Returns:
        list: List of session paths, or empty list if none found.
    """
    sessions = finder(path=path, pattern=r'\.(nidq|obx)\.meta$', multiple=True, folder=True)
    if sessions:
        print("\033[1;36mFound sessions to process with CatGT.\033[0m")
        return sessions
    print("No sessions found.")
    return []

def run_catgt(path=None):
    # Get CatGT executable
    catgt = get_file("catgt", "CatGT.exe", "CatGT")
    if not catgt:
        print("No CatGT executable found. Please select the CatGT.exe file.")
        return

    # Get user selections
    select_stream = inquirer.checkbox("Select stream types to process", 
                                    choices=['ob', 'ni', 'ap', 'lf'],
                                    default=['ob', 'ni', 'ap', 'lf'])
    if not select_stream:
        print("No streams selected. Exiting.")
        return

    select_all_gt = inquirer.confirm("Select all GT-ranges in the session", default=True)
    need_imec = any(s in select_stream for s in ['ap', 'lf'])
    select_all_imec = inquirer.confirm("Select all IMECs in the session", default=True) if need_imec else False

    # Get sessions to process
    sessions = get_sessions(path=path)
    if not sessions:
        return

    # Process each session
    cmds = []
    for session in sessions:
        session_name = os.path.basename(session)
        dest_dir = os.path.dirname(session)

        # Check for existing output
        if glob.glob(f"{dest_dir}/catgt_{session_name}*"):
            print(f"Skipping {session}: CatGT output already exists.")
            if not inquirer.confirm("Do you want to overwrite?", default=True):
                continue

        # Get GT ranges
        meta_files = glob.glob(f"{session}*/*.nidq.meta") + glob.glob(f"{session}*/*.obx.meta")
        session_gt = sorted(t.replace('.nidq.meta', '').replace('.obx.meta', '') for t in meta_files)
        selected_gt = session_gt if select_all_gt else inquirer.checkbox(
            message="Select sessions to merge",
            choices=session_gt,
            default=session_gt
        )

        # Format GT list for command
        gtlist = ''.join([
            "{" + re.search(r'_g(\d+)', gt).group(1) + "," + 
            re.search(r'_t(\d+)', gt).group(1) + "," + 
            re.search(r'_t(\d+)', gt).group(1) + "}" 
            for gt in selected_gt
        ])

        # Get probe info if needed
        i_imec = ''
        if need_imec:
            imecs = [folder for folder in glob.glob(f"{session}_g0/*imec?/")]
            imecs_id = [re.search(r'imec(\d+)', x).group(1) for x in imecs]
            if select_all_imec:
                i_imec = ','.join(imecs_id)
            else:
                selected_imecs = inquirer.checkbox(
                    message="Select imec probes to run CatGT",
                    choices=list(zip(imecs, imecs_id)),
                    default=list(zip(imecs, imecs_id))
                )
                i_imec = ','.join(selected_imecs)

        # Build commands for each stream type
        base_cmd = f'{catgt} -dir="{dest_dir}" -run="{session_name}" -gtlist={gtlist}'
        base_opts = f'-dest="{dest_dir}" -t_miss_ok -zerofillmax=0'

        for stream in select_stream:
            cmd = base_cmd
            if stream == 'ni' and glob.glob(f"{session}_g0\\*nidq.meta"):
                cmd += f' -ni {base_opts}'
            elif stream == 'ob':
                obx_files = glob.glob(f"{session}_g0\\*obx?.obx.meta")
                if obx_files:
                    obx_id = [re.search(r'obx(\d+)', x).group(1) for x in obx_files]
                    cmd += f' -ob -obx={",".join(obx_id)} {base_opts}'
                else:
                    continue
            elif stream == 'ap' and glob.glob(f"{session}_g0\\**\\*ap.meta"):
                cmd += f' -ap -prb={i_imec} {base_opts} -prb_fld -out_prb_fld -gbldmx'
            elif stream == 'lf' and glob.glob(f"{session}_g0\\**\\*lf.meta"):
                cmd += f' -lf -prb={i_imec} {base_opts} -prb_fld -out_prb_fld'
            else:
                continue
            cmds.append((session, stream, cmd))

    # Execute commands
    for i, (session, stream, cmd) in enumerate(cmds, 1):
        print(f"\033[91m{i}/{len(cmds)}: Running CatGT for {session} ({stream}): {cmd}\033[0m")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\033[91mError running CatGT for {session} ({stream}):\033[0m")
            print(result.stderr)
        else:
            print(f"\033[32mCatGT completed successfully for {session} ({stream})\033[0m")

if __name__ == '__main__':
    run_catgt(path='C:\\SGL_DATA')