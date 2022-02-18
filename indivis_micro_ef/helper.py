import math
import os
import sys
import pandas as pd
import numpy as np


def extr_input_dct(bwmhz, rho, N, seed, slot_duration):
    samp_rate = 1.0
    ss = 100
    num_server = 10
    mem_to_cpu_ratio = 1.0
    server_seed = seed
    channel_BW = bwmhz * 1.0e6
    realloc_overhead = 0.30
    num_day = 1
    scr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if sys.argv[-1] == 'large_cpu':
        c_lst = [[k, k] for k in [1.0, 0.75, 0.5, 0.25]] +\
                [[1.0 + k/10, (1.0 - k/10) * mem_to_cpu_ratio] for k in [9.9, 8.9, 7.9, 6.9]]
    elif sys.argv[-1] == 'large_mem':
        c_lst = [[k, k] for k in [1.0, 0.75, 0.5, 0.25]] +\
                [[1.0 - k/10, (1.0 + k/10) * mem_to_cpu_ratio] for k in [9.9, 8.9, 7.9, 6.9]]
    elif sys.argv[-1] == 'no_large':
        c_lst = [[k, k] for k in [1.0, 0.75, 0.5, 0.25]]
    else:
        c_lst = [[k, k] for k in [1.0, 0.75, 0.5, 0.25]] +\
                [[1.0 + k/10, (1.0 - k/10) * mem_to_cpu_ratio] for k in [9.9, 8.9, 7.9, 6.9]] +\
                [[1.0 - k/10, (1.0 + k/10) * mem_to_cpu_ratio] for k in [9.9, 8.9, 7.9, 6.9]]
#     df_mach = pd.DataFrame(np.array([[1.0 - rho/10, (1.0 + rho/10) * mem_to_cpu_ratio],
#                                      [1.0, 1.0 * mem_to_cpu_ratio],
#                                      [1.0 + rho/10, (1.0 - rho/10) * mem_to_cpu_ratio]]*10),
#                                     columns=['cap_cpu', 'cap_mem'])
    df_mach = pd.DataFrame(np.array(c_lst), columns=['cap_cpu', 'cap_mem'])
    dfm = df_mach.loc[:, ['cap_cpu', 'cap_mem']]
    helper_file_dir = os.path.dirname(__file__)
    google_input_dir = os.path.abspath(os.path.join(helper_file_dir, '..', 'google_input'))
    tot_task_event_df = pd.read_csv(os.path.join(google_input_dir, "1000_jobs.csv"),
                                    header=0, index_col=None)
    tot_task_event_df['job_id'] = tot_task_event_df.job_id.astype('int')
    N_large = int(float(sys.argv[4]) * N / 100)
    N_non_large = N - N_large
    input_df_non_large = tot_task_event_df.loc[tot_task_event_df.mice == 0].\
            sample(n=N_non_large, random_state=200+8*server_seed+1, replace=False, axis=0)
    input_df_large = tot_task_event_df.loc[tot_task_event_df.mice == 1]\
                     .sample(n=N_large, random_state=200+8*server_seed+1, replace=False, axis=0)
    input_df = pd.concat([input_df_non_large, input_df_large], axis=0, ignore_index=True)
    input_df.sort_values(by=['time_m'], inplace=True, ascending=True)
    input_df.reset_index(drop=True, inplace=True)
    input_df.time_m = input_df.time_m - input_df.iloc[0].time_m
    input_df.loc[:, f"sta_t"] = 0
    for i in range(0, len(dfm)):
        input_df.loc[:, f"sta_t"] = input_df[f"sta_t"] +\
                                    np.min([(dfm.to_numpy()[i, 0]/\
                                             input_df.cpu_requested).to_list(),
                                            (dfm.to_numpy()[i, 1]/\
                                             input_df.mem_requested).to_list()],
                                            axis=0).astype("int").tolist()
    input_df.loc[:, f"sta_t"] = np.min([(1/input_df[f"bw_requested_{int(bwmhz)}"])\
                                        .astype("int").to_list(),
                                        input_df[f"sta_t"].to_list()],
                                        axis=0).tolist()
    input_df.loc[:, f"sta_jct"] = (input_df["num_task"]/input_df[f"sta_t"])\
                                  .apply(lambda x: math.ceil(x)) * input_df["run_time_m"]
    input_df.rename(columns={'time_m': 'job_sub_time',
                             'num_task': 'int_num',
                             'cpu_requested': 'req_cpu_core',
                             'mem_requested': 'req_mem_GB',
                             'run_time_m': 'int_runtime_sec',
                             f"bw_requested_{int(bwmhz)}": 'req_bw'}, inplace=True)
    input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
                  'server_seed': server_seed, 'channel_BW': channel_BW,
                  'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
                  'scr_path': scr_path, 'slot_duration': slot_duration,
                  'realloc_overhead': realloc_overhead,
                  'num_day': num_day, 'rho': rho, 'N': N,
                  'input_df': input_df.loc[:, ['job_sub_time', 'job_id',
                                               'int_num', 'req_cpu_core',
                                               'req_mem_GB',
                                               'req_bw', 'int_runtime_sec',
                                               'mice', 'sta_t', 'sta_jct']]}
    return input_dict
