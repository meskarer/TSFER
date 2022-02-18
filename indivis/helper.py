import os
import sys
import pandas as pd
import numpy as np


def extr_input_dct(bwmhz, rho, N, seed, slot_duration):
    samp_rate = 1.0
    ss = 100
    num_server = 1
    mem_to_cpu_ratio = 1.0
    server_seed = seed
    channel_BW = bwmhz * 1.0e6
    realloc_overhead = 0.30
    num_day = 1
    scr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df_mach = pd.DataFrame(np.array([[1.0, 1.0 * mem_to_cpu_ratio],
                                     [1.0 - rho/10, (1.0 + rho/10) * mem_to_cpu_ratio],
                                     [1.0 + rho/10, (1.0 - rho/10) * mem_to_cpu_ratio]]),
                                    columns=['cap_cpu', 'cap_mem'])
    dfm = df_mach.loc[:, ['cap_cpu', 'cap_mem']]
    helper_file_dir = os.path.dirname(__file__)
    google_input_dir = os.path.abspath(os.path.join(helper_file_dir, '..', 'google_input'))
    tot_task_event_df = pd.read_csv(os.path.join(google_input_dir, "1000_jobs.csv"), header=0, index_col=None)
    tot_task_event_df['job_id'] = tot_task_event_df.job_id.astype('int')
    input_df = tot_task_event_df.sample(n=N, random_state=200+8*server_seed+1, replace=False, axis=0)
    input_df.sort_values(by=['time_m'], inplace=True, ascending=True)
    input_df.reset_index(drop=True, inplace=True)
    input_df.time_m = input_df.time_m - input_df.iloc[0].time_m
    input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
                  'server_seed': server_seed, 'channel_BW': channel_BW,
                  'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
                  'scr_path': scr_path, 'slot_duration': slot_duration,
                  'realloc_overhead': realloc_overhead,
                  'num_day': num_day, 'rho': rho, 'N': N,
                  'input_df': input_df}
    return input_dict
