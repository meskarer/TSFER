import sys
import numpy as np
from os import path, makedirs
from time import sleep
from random import randint
import pandas as pd


def read_argv(ch, ro, N, u_seed):
    sr = 1000
    samp_rate = sr / 1000.
    end_part = 1
    rho = ro / 10.0
    num_server = 1
    channel_bw = ch * 1.0e6
    mem_to_cpu_ratio = 1.0

    helper_file_dir = path.dirname(__file__)
    google_input_dir = path.abspath(path.join(helper_file_dir, '..', 'google_input'))
    tot_task_event_df = pd.read_csv(path.join(google_input_dir, "1000_jobs.csv"), header=0, index_col=None)
    tot_task_event_df['job_id'] = tot_task_event_df.job_id.astype('int')
    input_df = tot_task_event_df.sample(n=N, random_state=200+8*u_seed+1, replace=False, axis=0)
    input_df.sort_values(by=['time_m'], inplace=True, ascending=True)
    input_df.reset_index(drop=True, inplace=True)
    input_df.time_m = input_df.time_m - input_df.iloc[0].time_m

    c = np.array([[1.0, 1.0 * mem_to_cpu_ratio] for _ in range(num_server)] +
                 [[1.0 - rho, (1.0 + rho) * mem_to_cpu_ratio] for _ in range(num_server)] +
                 [[1.0 + rho, (1.0 - rho) * mem_to_cpu_ratio] for _ in range(num_server)])
    server_dict = {'c': c, 'num_server': 3*num_server, 'sampling_seed': u_seed,
                   'mem_to_cpu_ratio': mem_to_cpu_ratio, 'bw_MHz': ch, 'rho': rho}
    location_dict = {'trace_path': "/scratch/meskarer/clusterdata-2011-2",
                     'scratch_path': "/scratch/meskarer"}
    event_reading_dict = {'jump': 1 , 'start_part': 1 , 'end_part': end_part,
                          'sample_rate': samp_rate, 'sampling_seed': u_seed,
                          'input_df': input_df}
    scr_path = path.abspath(path.join(path.dirname(__file__), '..'))
    RW_path = path.join(scr_path, 'google_divis', f"{u_seed}sampling_seed")
    location_dict['RW_path'] = RW_path

    return location_dict, event_reading_dict, server_dict, channel_bw
