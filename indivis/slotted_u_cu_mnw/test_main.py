from waiting_log import Utilitarian_waiting_log, Cautious_utilitarian_waiting_log, MNW_waiting_log
from running_log import Running_log
from finished_log import Finished_log
from env import Slotted_utilitarian_env, Slotted_cautious_utilitarian_env, Slotted_mnw_env
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
            slot_duration = 100
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
    
    # u_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
    #                 'server_seed': server_seed, 'channel_BW': channel_BW,
    #                 'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
    #                 'scr_path': scr_path, 'slot_duration': slot_duration,
    #                 'realloc_overhead': realloc_overhead}
    # u_waiting_log = Utilitarian_waiting_log()
    # u_running_log = Running_log()
    # u_finished_log = Finished_log()
    # u_env = Slotted_utilitarian_env(u_input_dict, u_waiting_log,
    #                                 u_running_log, u_finished_log)
    # u_env.run()
    # u_env.save_finished_log()
    # u_env.save_df_event()

    # cu_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
    #                  'server_seed': server_seed, 'channel_BW': channel_BW,
    #                  'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
    #                  'scr_path': scr_path, 'slot_duration': slot_duration,
    #                  'realloc_overhead': realloc_overhead}
    # cu_waiting_log = Cautious_utilitarian_waiting_log()
    # cu_running_log = Running_log()
    # cu_finished_log = Finished_log()
    # cu_env = Slotted_cautious_utilitarian_env(cu_input_dict, 
    #                                           cu_waiting_log,
    #                                           cu_running_log,
    #                                           cu_finished_log)
    # cu_env.run()
    # cu_env.save_finished_log()
    # cu_env.save_df_event()

    mnw_input_dict = {'samp_rate': samp_rate, 'num_server': num_server,
                      'server_seed': server_seed, 'channel_BW': channel_BW,
                      'extr_res_cap': 1.0, 'comp_res_cap': dfm.to_numpy(),
                      'scr_path': scr_path, 'slot_duration': slot_duration,
                      'realloc_overhead': realloc_overhead}
    mnw_waiting_log = MNW_waiting_log()
    mnw_running_log = Running_log()
    mnw_finished_log = Finished_log()
    mnw_env = Slotted_mnw_env(mnw_input_dict, 
                              mnw_waiting_log,
                              mnw_running_log,
                              mnw_finished_log)
    mnw_env.run()
    mnw_env.save_finished_log()
    mnw_env.save_df_event()