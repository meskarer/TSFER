import math
import os
import sys
import pandas as pd
import numpy as np
SPEC_EFFICIENCY = 3.5


def extr_input_dct(bwmhz, rho, N, seed, slot_duration):
    is_google = True if sys.argv[3] == 'g' else False
    is_alibaba = True if sys.argv[3] == 'a' else False
    samp_rate = 1.0
    num_server = 1 if is_google else 20
    mem_to_cpu_ratio = 1.0
    server_seed = seed
    channel_BW = bwmhz * 1.0e6
    realloc_overhead = 0.30
    num_day = 1
    scr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    helper_file_dir = os.path.dirname(__file__)
    if is_google:
        df_mach = pd.DataFrame(np.array([[1.0, 1.0 * mem_to_cpu_ratio],
                                         [1.0 - rho/10, (1.0 + rho/10) * mem_to_cpu_ratio],
                                         [1.0 + rho/10, (1.0 - rho/10) * mem_to_cpu_ratio]]),
                                        columns=['cap_cpu', 'cap_mem'])
        dfm = df_mach.loc[:, ['cap_cpu', 'cap_mem']]
        google_input_dir = os.path.abspath(os.path.join(helper_file_dir, '..',
                                                        'google_input'))
        tot_task_event_df = pd.read_csv(os.path.join(google_input_dir, "1000_jobs.csv"),
                                        header=0, index_col=None)
        tot_task_event_df['job_id'] = tot_task_event_df.job_id.astype('int')
        input_df = tot_task_event_df.sample(n=N, random_state=200+8*server_seed+1,
                                            replace=False, axis=0)
        input_df.sort_values(by=['time_m'], inplace=True, ascending=True)
        input_df.reset_index(drop=True, inplace=True)
        input_df.time_m = input_df.time_m - input_df.iloc[0].time_m
        input_df.rename(columns={'time_m': 'job_sub_time',
                                 'num_task': 'int_num',
                                 'cpu_requested': 'req_cpu_core',
                                 'mem_requested': 'req_mem_GB',
                                 'run_time_m': 'int_runtime_sec',
                                 f"bw_requested_{int(bwmhz)}": 'req_bw',
                                 f"indiv_sta_t_bw{int(bwmhz)}_rho{rho}": 'sta_t',
                                 f"indiv_sta_jct_bw{int(bwmhz)}_rho{rho}": 'sta_jct'},
                                 inplace=True)
    elif is_alibaba:
        mach_path = os.path.join(scr_path, 'alibaba_trace', 'mach_data',
                                 f"ali_mach_numserv{num_server}_seed{server_seed}.csv")
        df_mach = pd.read_csv(mach_path, header=0, index_col=0,
                              dtype={'cap_cpu': 'float64',
                                     'cap_mem': 'float64',
                                     'cap_gpu': 'float64',
                                     'machine': 'str'})
        dfm = df_mach.loc[:, ['cap_cpu', 'cap_mem', 'cap_gpu']]
        task_event_folder = os.path.abspath(os.path.join(helper_file_dir, '..', 'alibaba_trace',
                                                     'input'))
        day = int(sys.argv[4])
        rate = int(sys.argv[5])
        task_event_n = f'{day}_sampled{rate}_event_nonzero.csv'
        task_event_path = os.path.join(task_event_folder, task_event_n)
        input_df = pd.read_csv(task_event_path, header=0, index_col=0,
                               dtype={'job_sub_time': 'float64',
                                      'job_id': 'str',
                                      'int_num': 'float64',
                                      'req_cpu_core': 'float64',
                                      'req_mem_GB': 'float64',
                                      'req_gpu': 'float64',
                                      'req_bw': 'float64',
                                      'int_runtime_sec': 'float64'})
        spec_efficiency = 3.5
        input_df['req_bit_per_sec'] = (input_df['req_bit_per_sec'] /
                                       SPEC_EFFICIENCY / channel_BW)
        input_df.rename(columns={'req_bit_per_sec': 'req_bw'}, inplace=True)
        # Find sta_t and jct and mice
        input_df['sta_t'] = 0
        input_df['sta_jct'] = 0
        input_df.drop(index=input_df.loc[input_df.int_runtime_sec == 0].index, inplace=True)
        for row in range(20):
            input_df.loc[:, 'sta_t'] = input_df['sta_t'] +\
                np.min([(dfm.iloc[row]['cap_cpu']/input_df.req_cpu_core).tolist(),
                        (dfm.iloc[row]['cap_mem']/input_df.req_mem_GB).tolist(),
                        (dfm.iloc[row]['cap_gpu']/input_df.req_gpu).tolist()], axis=0).astype("int").tolist()
        input_df.loc[:, 'sta_t'] = np.min([(1/input_df['req_bw']).astype("int").tolist(),
                                           input_df['sta_t'].tolist()], axis=0).tolist()
        input_df.loc[:, 'sta_jct'] = (input_df['int_num']/input_df['sta_t']).apply(lambda x: math.ceil(x)) * input_df['int_runtime_sec']
        thresh = [np.quantile(input_df.sta_jct, 0.5), np.quantile(input_df.sta_jct, 0.95)]
        input_df['mice'] = input_df.sta_jct.transform(lambda x: 0 if x <= thresh[0] else 1 if x <= thresh[1] else 2)
    input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
                  'server_seed': server_seed, 'channel_BW': channel_BW,
                  'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
                  'scr_path': scr_path, 'slot_duration': slot_duration,
                  'realloc_overhead': realloc_overhead,
                  'num_day': num_day, 'rho': rho, 'N': N,
                  'input_df': input_df}
    return input_dict
