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

from .utils import finder, get_file, file_reorder

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

def run_catgt(path=None, supercat=False):
    # Get CatGT executable
    catgt = get_file("catgt", "CatGT.exe", "CatGT")
    if not catgt:
        print("No CatGT executable found. Please select the CatGT.exe file.")
        return

    # Get user selections
    select_stream = inquirer.checkbox("Select stream types to process", 
                                    choices=['ob', 'ni', 'ap', 'lf'],
                                    default=['ob', 'ni', 'ap'])
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

    if len(sessions) == 1:
        supercat = False

    # Reorder sessions
    if supercat and len(sessions) > 1:
        sessions = file_reorder(sessions)

    # Process each session
    cmds = []
    for session in sessions:
        session_name = os.path.basename(session).replace('catgt_', '')
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
        base_opts = f' -dest="{dest_dir}" -t_miss_ok -zerofillmax=0'

        cmd = base_cmd
        if 'ni' in select_stream and glob.glob(f"{session}_g0\\*nidq.meta"):
            cmd += f' -ni -pass1_force_ni_ob_bin'
        if 'ob' in select_stream and glob.glob(f"{session}_g0\\*obx?.obx.meta"):
            obx_files = glob.glob(f"{session}_g0\\*obx?.obx.meta")
            if obx_files:
                obx_id = [re.search(r'obx(\d+)', x).group(1) for x in obx_files]
                cmd += f' -ob -obx={",".join(obx_id)} -pass1_force_ni_ob_bin'
        if 'ap' in select_stream and glob.glob(f"{session}_g0\\**\\*ap.meta"):
            cmd += f' -ap -prb={i_imec} -prb_fld -out_prb_fld -gbldmx -apfilter=butter,12,300,9000 -gfix=0.40,0.10,0.02'
        if 'lf' in select_stream and glob.glob(f"{session}_g0\\**\\*lf.meta"):
            cmd += f' -lf'
            if 'ap' not in select_stream:
                cmd += f' -prb={i_imec} -prb_fld -out_prb_fld'
        cmd += base_opts
        cmds.append((session, cmd))

    # Execute commands
    for i, (session, cmd) in enumerate(cmds, 1):
        print(f"\033[91m{i}/{len(cmds)}: Running CatGT for {session}: {cmd}\033[0m")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\033[91mError running CatGT for {session}:\033[0m")
            print(result.stderr)
        else:
            print(f"\033[32mCatGT completed successfully for {session}\033[0m")

    if supercat:
        cmds = []
        supercat_dir = []
        dest_dir0 = None
        session_name0 = None
        for session in sessions:
            dest_dir = os.path.dirname(session)
            session_name = os.path.basename(session).replace('catgt_', '')
            cat_fn = glob.glob(f"{dest_dir}/catgt_{session_name}*")
            if cat_fn:
                cat_name = os.path.basename(cat_fn[0])
                supercat_dir.append(f'{{{dest_dir},{cat_name}}}')
                if dest_dir0 is None:
                    dest_dir0 = dest_dir
                    session_name0 = cat_name.replace('catgt_', '')
        
        if not supercat_dir:
            print("No CatGT output found. Exiting.")
            return
        supercat_dir = ''.join(supercat_dir)

        # Check for existing output
        if glob.glob(f"{dest_dir0}/supercat_{session_name0}*"):
            print(f"CatGT Supercat output already exists.")
            if not inquirer.confirm("Do you want to overwrite?", default=True):
                return

        # Get probe info if needed
        i_imec = ''
        if need_imec:
            imecs = [folder for folder in glob.glob(f"{dest_dir0}/catgt_{session_name0}/*imec?/")]
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
        base_cmd = f'{catgt} -supercat="{supercat_dir}"'
        base_opts = f' -dest="{dest_dir0}"'

        cmd = base_cmd
        if 'ni' in select_stream and glob.glob(f"{dest_dir0}\\catgt_{session_name0}\\*nidq.meta"):
            cmd += f' -ni'
        if 'ob' in select_stream and glob.glob(f"{dest_dir0}\\catgt_{session_name0}\\*obx?.obx.meta"):
            obx_files = glob.glob(f"{dest_dir0}\\catgt_{session_name0}\\*obx?.obx.meta")
            if obx_files:
                obx_id = [re.search(r'obx(\d+)', x).group(1) for x in obx_files]
                cmd += f' -ob -obx={",".join(obx_id)}'
        if 'ap' in select_stream and glob.glob(f"{dest_dir0}\\catgt_{session_name0}\\**\\*ap.meta"):
            cmd += f' -ap -prb={i_imec} -prb_fld -out_prb_fld'
        if 'lf' in select_stream and glob.glob(f"{dest_dir0}\\catgt_{session_name0}\\**\\*lf.meta"):
            cmd += f' -lf'
            if 'ap' not in select_stream:
                cmd += f' -prb={i_imec} -prb_fld -out_prb_fld'
        cmd += base_opts
        cmds.append((session_name0, cmd))

        # Execute commands
        for i, (stream, cmd) in enumerate(cmds, 1):
            print(f"\033[91m{i}/{len(cmds)}: Running CatGT for {stream}: {cmd}\033[0m")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"\033[91mError running CatGT for {stream}:\033[0m")
                print(result.stderr)
            else:
                print(f"\033[32mCatGT completed successfully ({stream})\033[0m")

if __name__ == '__main__':
    run_catgt(path='C:\\SGL_DATA', supercat=True)