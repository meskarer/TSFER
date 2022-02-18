from helper import extr_input_dct
from slotted_egaliterian.waiting_log import TSFER_waiting_log, DRFER_waiting_log
from slotted_egaliterian.running_log import Running_log as Running_log_egal
from slotted_egaliterian.finished_log import Finished_log as Finished_log_egal
from slotted_egaliterian.env import Slotted_TSFER_env, Slotted_DRFER_env
from slotted_u_cu_mnw.waiting_log import Cautious_relative_utilitarian_waiting_log, MNW_waiting_log
from slotted_u_cu_mnw.running_log import Running_log as Running_log_u
from slotted_u_cu_mnw.finished_log import Finished_log as Finished_log_u
from slotted_u_cu_mnw.env import Slotted_cautious_relative_utilitarian_env, Slotted_mnw_env
import os
import sys
import pandas as pd
import numpy as np
import pickle as pk


def empty_df():
    empti_df = pd.DataFrame({'job_id': pd.Series(dtype='int64'),
                             'n': pd.Series(dtype='int'),
                             'bw': pd.Series(dtype='int'),
                             'seed': pd.Series(dtype='int'),
                             'slot_dur_s': pd.Series(dtype='int'),
                             'rho': pd.Series(dtype='int'),
                             'mice': pd.Series(dtype='int'),
                             'boost': pd.Series(dtype='float')})
    return empti_df


if __name__ == "__main__":
    # arg 1: seed start (inclusive)
    # arg 2: seed end (inclusive)
    # arg 3: slot duration in second
    # seed_start = int(sys.argv[1])
    # seed_end = int(sys.argv[2])
    seed_start = 100
    seed_end = 100
    bw_list = [3]
    rho_list = [6]
    nu_list = [4]
    # slot_dur_list = [1.0, 4.0, 16.0, 64.0, 256.0]
    slot_dur_list = [120.0]
#     bw_list = [2]
#     rho_list = [9]
#     nu_list = [10]
#     slot_dur_list = [1.0]
    tsfer_boost_df = empty_df()
    tsfer_boost_df = empty_df()
    drfer_boost_df = empty_df()
    mnw_boost_df = empty_df()
    cru_boost_df = empty_df()

    for slot_duration in slot_dur_list:
        for seed in range(seed_start, seed_end+1):
            for n_u in nu_list:
                for bwmhz in bw_list:
                    for rho in rho_list:
                        print(f"slot: {slot_duration}, seed: {seed}, n_u: {n_u}, bwmhz: {bwmhz}, rho: {rho}")
                        input_dict = extr_input_dct(bwmhz, rho, n_u, seed, slot_duration)
                        tsfer_waiting_log = TSFER_waiting_log()
                        tsfer_running_log = Running_log_egal()
                        tsfer_finished_log = Finished_log_egal()
                        tsfer_env = Slotted_TSFER_env(input_dict,
                                                      tsfer_waiting_log,
                                                      tsfer_running_log,
                                                      tsfer_finished_log)
                        num_task_dct = tsfer_env.run()
                        init_df, job_df = tsfer_env.save_finished_log()
                        # tsfer_env.event_record.to_csv(f"simulation/tsfer_{int(slot_duration)}_event_log.csv", header=True)
                        # init_df.to_csv(f"simulation/tsfer_{int(slot_duration)}.csv", header=True)
                        tmp_df = empty_df()
                        tmp_df['boost'] = pd.Series(job_df['boost'], dtype='float')
                        tmp_df['job_id'] = pd.Series(job_df['job_id'], dtype='int')
                        tmp_df['mice'] = pd.Series(job_df['mice'], dtype='int')
                        tmp_df.loc[:, 'n'] = n_u
                        tmp_df.loc[:, 'bw'] = bwmhz
                        tmp_df.loc[:, 'rho'] = rho
                        tmp_df.loc[:, 'seed'] = seed
                        tmp_df.loc[:, 'slot_dur_s'] = int(slot_duration)
                        tsfer_boost_df = tsfer_boost_df.append(tmp_df, ignore_index=True)
                        del(tmp_df)
                        time_lst = []
                        event_lst = []
                        rate = {1: [], 2: [], 3: [], 4: []}
                        for k, v in num_task_dct.items():
                            time_lst.append(k)
                            event_lst.append(v['event'])
                            for job_id in [1, 2, 3, 4]:
                                if job_id not in v:
                                    rate[job_id].append(0)
                                else:
                                    rate[job_id].append(v[job_id])
                        pk.dump({'time': time_lst,
                                 'event': event_lst,
                                 'rate': rate},
                                 open(f"simulation/tsfer_{int(slot_duration)}.pk", "wb"))

                        drfer_waiting_log = DRFER_waiting_log()
                        drfer_running_log = Running_log_egal()
                        drfer_finished_log = Finished_log_egal()
                        drfer_env = Slotted_DRFER_env(input_dict,
                                                      drfer_waiting_log,
                                                      drfer_running_log,
                                                      drfer_finished_log)
                        num_task_dct = drfer_env.run()
                        init_df, job_df = drfer_env.save_finished_log()
                        # drfer_env.event_record.to_csv(f"simulation/drfer_{int(slot_duration)}_event_log.csv", header=True)
                        # init_df.to_csv(f"simulation/drfer_{int(slot_duration)}.csv", header=True)
                        tmp_df = empty_df()
                        tmp_df['boost'] = pd.Series(job_df['boost'], dtype='float')
                        tmp_df['job_id'] = pd.Series(job_df['job_id'], dtype='int')
                        tmp_df['mice'] = pd.Series(job_df['mice'], dtype='int')
                        tmp_df.loc[:, 'n'] = n_u
                        tmp_df.loc[:, 'bw'] = bwmhz
                        tmp_df.loc[:, 'rho'] = rho
                        tmp_df.loc[:, 'seed'] = seed
                        tmp_df.loc[:, 'slot_dur_s'] = int(slot_duration)
                        drfer_boost_df = drfer_boost_df.append(tmp_df, ignore_index=True)
                        del(tmp_df)
                        time_lst = []
                        event_lst = []
                        rate = {1: [], 2: [], 3: [], 4: []}
                        for k, v in num_task_dct.items():
                            time_lst.append(k)
                            event_lst.append(v['event'])
                            for job_id in [1, 2, 3, 4]:
                                if job_id not in v:
                                    rate[job_id].append(0)
                                else:
                                    rate[job_id].append(v[job_id])
                        pk.dump({'time': time_lst,
                                 'event': event_lst,
                                 'rate': rate},
                                 open(f"simulation/drfer_{int(slot_duration)}.pk", "wb"))
                        
                        mnw_waiting_log = MNW_waiting_log()
                        mnw_running_log = Running_log_u()
                        mnw_finished_log = Finished_log_u()
                        mnw_env = Slotted_mnw_env(input_dict, 
                                                  mnw_waiting_log,
                                                  mnw_running_log,
                                                  mnw_finished_log)
                        num_task_dct = mnw_env.run()
                        init_df, job_df = mnw_env.save_finished_log()
                        # mnw_env.event_record.to_csv(f"simulation/mnw_{int(slot_duration)}_event_log.csv", header=True)
                        # init_df.to_csv(f"simulation/mnw_{int(slot_duration)}.csv", header=True)
                        tmp_df = empty_df()
                        tmp_df['boost'] = pd.Series(job_df['boost'], dtype='float')
                        tmp_df['job_id'] = pd.Series(job_df['job_id'], dtype='int')
                        tmp_df['mice'] = pd.Series(job_df['mice'], dtype='int')
                        tmp_df.loc[:, 'n'] = n_u
                        tmp_df.loc[:, 'bw'] = bwmhz
                        tmp_df.loc[:, 'rho'] = rho
                        tmp_df.loc[:, 'seed'] = seed
                        tmp_df.loc[:, 'slot_dur_s'] = int(slot_duration)
                        mnw_boost_df = mnw_boost_df.append(tmp_df, ignore_index=True)
                        del(tmp_df)
                        time_lst = []
                        event_lst = []
                        rate = {1: [], 2: [], 3: [], 4: []}
                        for k, v in num_task_dct.items():
                            time_lst.append(k)
                            event_lst.append(v['event'])
                            for job_id in [1, 2, 3, 4]:
                                if job_id not in v:
                                    rate[job_id].append(0)
                                else:
                                    rate[job_id].append(v[job_id])
                        pk.dump({'time': time_lst,
                                 'event': event_lst,
                                 'rate': rate},
                                 open(f"simulation/mnw_{int(slot_duration)}.pk", "wb"))

                        cru_waiting_log = Cautious_relative_utilitarian_waiting_log()
                        cru_running_log = Running_log_u()
                        cru_finished_log = Finished_log_u()
                        cru_env = \
                            Slotted_cautious_relative_utilitarian_env(input_dict, 
                                                                      cru_waiting_log,
                                                                      cru_running_log,
                                                                      cru_finished_log)
                        num_task_dct = cru_env.run()
                        init_df, job_df = cru_env.save_finished_log()     
                        # cru_env.event_record.to_csv(f"simulation/cru_{int(slot_duration)}_event_log.csv", header=True)
                        # init_df.to_csv(f"simulation/cru_{int(slot_duration)}.csv", header=True)
                        tmp_df = empty_df()
                        tmp_df['boost'] = pd.Series(job_df['boost'], dtype='float')
                        tmp_df['job_id'] = pd.Series(job_df['job_id'], dtype='int')
                        tmp_df['mice'] = pd.Series(job_df['mice'], dtype='int')
                        tmp_df.loc[:, 'n'] = n_u
                        tmp_df.loc[:, 'bw'] = bwmhz
                        tmp_df.loc[:, 'rho'] = rho
                        tmp_df.loc[:, 'seed'] = seed
                        tmp_df.loc[:, 'slot_dur_s'] = int(slot_duration)
                        cru_boost_df = cru_boost_df.append(tmp_df, ignore_index=True)
                        del(tmp_df)
                        time_lst = []
                        event_lst = []
                        rate = {1: [], 2: [], 3: [], 4: []}
                        for k, v in num_task_dct.items():
                            time_lst.append(k)
                            event_lst.append(v['event'])
                            for job_id in [1, 2, 3, 4]:
                                if job_id not in v:
                                    rate[job_id].append(0)
                                else:
                                    rate[job_id].append(v[job_id])
                        pk.dump({'time': time_lst,
                                 'event': event_lst,
                                 'rate': rate},
                                 open(f"simulation/cru_{int(slot_duration)}.pk", "wb"))

    j_l = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'google_micro_indivis'))
    # tsfer_boost_df.to_csv(os.path.join(j_l, f"indivis_tsfer_boost_seed{seed_start}-{seed_end}.csv.gz"))
    # drfer_boost_df.to_csv(os.path.join(j_l, f"indivis_drfer_boost_seed{seed_start}-{seed_end}.csv.gz"))
    # mnw_boost_df.to_csv(os.path.join(j_l, f"indivis_mnw_boost_seed{seed_start}-{seed_end}.csv.gz"))
    # cru_boost_df.to_csv(os.path.join(j_l, f"indivis_cru_boost_seed{seed_start}-{seed_end}.csv.gz"))