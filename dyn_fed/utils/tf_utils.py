"""Tensorflow utility functions
"""
from pathlib import Path

import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def _load_event(path, keys):
    """Load events from a single folder
    """
    # Accumulate events
    x = EventAccumulator(path=path)
    x.Reload()
    # x.FirstEventTimestamp() 

    try:
        steps = [e.step for e in x.Scalars(keys[0])]
        wall_time = [e.wall_time for e in x.Scalars(keys[0])]
        index = [e.index for e in x.Scalars(keys[0])]
        count = [e.count for e in x.Scalars(keys[0])]
    except KeyError:
        print(f"Skipping {path}")
        return
    n_steps = len(steps)
    list_run = [Path(path).parents[1].name] * n_steps

    data = np.zeros((n_steps, len(keys)))
    for i in range(len(keys)):
        data[:, i] = [e.value for e in x.Scalars(keys[i])]

    # printOutDict = {
    #     keys[0]: data[:,0],
    #     keys[1]: data[:,1],
    #     keys[2]: data[:,2],
    #     keys[3]: data[:,3]
    # }
    print_out_dict = {keys[i]: data[:, i] for i in range(len(keys))}
    print_out_dict['Name'] = list_run

    df = pd.DataFrame(data=print_out_dict)

    return df

def load_events(folders, keys, child_dir=None):
    """Load tensorflow events
    """
    df_list = []
    for tb_output_folder in folders:
        if child_dir is not None:
            tb_output_folder = str(Path(tb_output_folder)/f'{child_dir}')
        df = _load_event(tb_output_folder, keys=keys)
        df_list.append(df)

    # Concatenate all dfs
    dfs = pd.concat(df_list)

    return dfs
