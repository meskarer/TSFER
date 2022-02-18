from waiting_log import TSFER_waiting_log, DRFER_waiting_log,\
    Naive_TSF_waiting_log, Naive_DRF_waiting_log
from running_log import Running_log
from finished_log import Finished_log
from env import Slotted_TSFER_env, Slotted_DRFER_env,\
    Slotted_Naive_TSF_env, Slotted_Naive_DRF_env
import os
import sys
import pandas as pd
import numpy as np


if __name__ == "__main__":
    if os.popen('hostname').read()[0:-1] not in ['erfan-Inspiron-3670',
                                                 'DESKTOP-VVQRRIT']:
        samp_rate = float(sys.argv[1]) / 100.
        num_server = int(sys.argv[2])
        server_seed = int(sys.argv[3])
        channel_BW = float(sys.argv[4]) * 1e6
        slot_duration = float(sys.argv[5])
        realloc_overhead = float(sys.argv[6])
        if os.path.isdir('/scratch/b/benliang/meskarer'):
            scr_path = os.path.join('/scratch/b/benliang/meskarer')
        elif os.path.isdir('/scratch/meskarer'):
            scr_path = os.path.join('/scratch/meskarer')
    else:
        if len(sys.argv) > 1:
            samp_rate = float(sys.argv[1]) / 100.
            num_server = int(sys.argv[2])
            server_seed = int(sys.argv[3])
            channel_BW = float(sys.argv[4]) * 1e6
            slot_duration = float(sys.argv[5])
            realloc_overhead = float(sys.argv[6])
        else:
            samp_rate = 1. / 100.
            num_server = 20
            server_seed = 23432
            channel_BW = 6e9
            slot_duration = 1800
            realloc_overhead = 0.25
        if os.path.isdir('/home/erfan/Projects/PhD/external-resource/TSF-ER-2/code-python/TON'):
            scr_path = os.path.join('/home/erfan/Projects/PhD/external-resource/TSF-ER-2/code-python/TON')
        elif os.path.isdir(os.path.join('D:\Projects', 'TON_OOP_V03')):
            scr_path = os.path.join('D:\Projects', 'TON_OOP_V03')
        else:
            raise Exception("Cannot find appropariate address to read and write data.")
    mach_path = os.path.join(scr_path, 'alibaba_trace', 'mach_data',
                             f"ali_mach_numserv{num_server}_seed{server_seed}.csv")
    df_mach = pd.read_csv(mach_path, dtype={'cap_cpu': 'float64',
                                            'cap_mem': 'float64',
                                            'cap_gpu': 'float64',
                                            'machine': 'str'})
    dfm = df_mach.loc[:, ['cap_cpu', 'cap_mem', 'cap_gpu']] 
    
    tsfer_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
                        'server_seed': server_seed, 'channel_BW': channel_BW,
                        'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
                        'scr_path': scr_path, 'slot_duration': slot_duration,
                        'realloc_overhead': realloc_overhead}
    tsfer_waiting_log = TSFER_waiting_log()
    tsfer_running_log = Running_log()
    tsfer_finished_log = Finished_log()
    tsfer_env = Slotted_TSFER_env(tsfer_input_dict, tsfer_waiting_log,
                                  tsfer_running_log, tsfer_finished_log)
    tsfer_env.run()
    tsfer_env.save_finished_log()
    tsfer_env.save_df_event()

    # drfer_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
    #                     'server_seed': server_seed, 'channel_BW': channel_BW,
    #                     'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
    #                     'scr_path': scr_path, 'slot_duration': slot_duration,
    #                     'realloc_overhead': 0.1}
    # drfer_waiting_log = DRFER_waiting_log()
    # drfer_running_log = Running_log()
    # drfer_finished_log = Finished_log()
    # drfer_env = Slotted_DRFER_env(drfer_input_dict, drfer_waiting_log,
    #                               drfer_running_log, drfer_finished_log)
    # drfer_env.run()
    # drfer_env.save_finished_log()
    # drfer_env.save_df_event()
    
    # naive_tsf_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
    #                         'server_seed': server_seed,
    #                         'channel_BW': channel_BW,
    #                         'extr_res_cap': np.ones(shape=(20, 1)) / 20,
    #                         'comp_res_cap': dfm.to_numpy(),
    #                         'scr_path': scr_path,
    #                         'slot_duration': slot_duration,
    #                         'realloc_overhead': 0.1}
    # naive_tsf_waiting_log = Naive_TSF_waiting_log()
    # naive_tsf_running_log = Running_log()
    # naive_tsf_finished_log = Finished_log()
    # naive_tsf_env = Slotted_Naive_TSF_env(naive_tsf_input_dict,
    #                                       naive_tsf_waiting_log,
    #                                       naive_tsf_running_log,
    #                                       naive_tsf_finished_log)
    # naive_tsf_env.run()
    # naive_tsf_env.save_finished_log()
    # naive_tsf_env.save_df_event()

    # naive_drf_input_dict = {'samp_rate': samp_rate, 'num_server': num_server, 
    #                         'server_seed': server_seed,
    #                         'channel_BW': channel_BW,
    #                         'extr_res_cap': np.ones(shape=(20, 1)) / 20,
    #                         'comp_res_cap': dfm.to_numpy(),
    #                         'scr_path': scr_path,
    #                         'slot_duration': slot_duration,
    #                         'realloc_overhead': 0.1}
    # naive_drf_waiting_log = Naive_DRF_waiting_log()
    # naive_drf_running_log = Running_log()
    # naive_drf_finished_log = Finished_log()
    # naive_drf_env = Slotted_Naive_DRF_env(naive_drf_input_dict,
    #                                       naive_drf_waiting_log,
    #                                       naive_drf_running_log,
    #                                       naive_drf_finished_log)
    # naive_drf_env.run()
    # naive_drf_env.save_finished_log()
    # naive_drf_env.save_df_event()