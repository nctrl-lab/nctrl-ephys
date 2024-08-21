# catgt.py
#
# This module provides functionality for working with CatGT, a tool for processing
# electrophysiology data. It includes functions for selecting the CatGT executable,
# managing file paths, and processing session data.
# 
# * By default, we will ignore real-world time gaps (zerofillmax=0).
#
# Download CatGT from https://billkarsh.github.io/SpikeGLX/
#
# SpikeGLX records files in the following order:
#   g0_t0, g0_t1, g0_t2, ...
#   g1_t0, g1_t1, g1_t2, ...
#   g2_t0, g2_t1, g2_t2, ...


import re
import glob
import os
import inquirer
import keyring
import subprocess

import tkinter as tk
from tkinter import filedialog

from .utils import finder

def get_catgt():
    fn = keyring.get_password("nctrl", "catgt")
    if fn and os.path.exists(fn):
        print(f"Using CatGT executable: {fn}")
        return fn

    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    print("\033[1;36mSelect the CatGT executable\033[0m")
    fn = filedialog.askopenfilename(
        title="Select the CatGT executable",
        filetypes=[("Executable file", "*.exe")]
    )
    root.destroy()

    if fn and os.path.exists(fn) and os.path.basename(fn) == 'CatGT.exe':
        keyring.set_password("nctrl", "catgt", fn)
        return fn
    else:
        print("Invalid selection. Please choose the CatGT.exe file.")
        return None

def get_sessions(path=None):
    print("\033[1;36mPlease select the sessions to process using CatGT.\033[0m")
    sessions = finder(path=path, pattern=r'\.nidq\.meta$', multiple=True, folder=True)
    if sessions is None:
        return []
    sessions = list(set([re.sub(r'_g\d+$', '', i) for i in sessions]))
    if not sessions:
        print("No sessions found.")
        return []
    return sessions

def run_catgt(path=None):
    catgt = get_catgt()
    if not catgt:
        print("No CatGT executable found. Please select the CatGT.exe file.")
        return
    
    # Options
    select_stream = inquirer.checkbox("Select stream types to process", choices=['ni', 'ap', 'lf'], default=['ni', 'ap', 'lf'])
    if not select_stream:
        print("No streams selected. Exiting.")
        return

    select_all_gt = inquirer.confirm("Select all GT-ranges in the session", default=True)
    select_all_imec = inquirer.confirm("Select all IMECs in the session", default=True) if ('ap' in select_stream or 'lf' in select_stream) else False
    
    sessions = get_sessions(path=path)
    if not sessions:
        return
    
    cmds = []
    for session in sessions:
        dest_dir = f"{session}_g0"
        if glob.glob(f"{dest_dir}/catgt_*/"):
            print(f"Skipping {session}: CatGT output already exists.")
            if not inquirer.confirm("Do you want to overwrite?", default=True):
                continue
        
        # Select Gate and Time (T)
        session_gt = sorted([t.replace('.nidq.meta', '') for t in glob.glob(f"{session}*/*.nidq.meta")])
        selected_gt = session_gt if select_all_gt else inquirer.checkbox(
            message="Select sessions to merge",
            choices=session_gt,
            default=session_gt
        )

        gtlist = ''.join(["{g" + re.search(r'_g(\d+)', gt).group(1) + ",t" + re.search(r'_t(\d+)', gt).group(1) + "a,t" + re.search(r'_t(\d+)', gt).group(1) + "b}" for gt in selected_gt])

        # Select Probe
        i_imec = ''
        if 'ap' in select_stream or 'lf' in select_stream:
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
        
        session_name = os.path.basename(session)
        animal_dir = os.path.dirname(session)
        
        for stream in select_stream:
            cmd = f'{catgt} -dir="{animal_dir}" -run="{session_name}" -gtlist={gtlist}'
            if stream == 'ni':
                # run CatGT only when multiple GTs are selected
                # if len(selected_gt) < 2:
                #     continue
                cmd += f' -ni -dest="{dest_dir}" -t_miss_ok -zerofillmax=0'
            elif stream == 'ap':
                cmd += f' -ap -prb={i_imec} -dest="{dest_dir}" -prb_fld -t_miss_ok -zerofillmax=0 -out_prb_fld -gbldmx'
            elif stream == 'lf':
                # make sure lf.meta exists
                if not glob.glob(f"{session}_g0\\**\\*lf.meta"):
                    continue
                # if len(selected_gt) < 2:
                #     continue
                cmd += f' -lf -prb={i_imec} -dest="{dest_dir}" -prb_fld -t_miss_ok -zerofillmax=0 -out_prb_fld'
            cmds.append((session, stream, cmd))
    
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