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


class Boost_df:
    def __init__(self):
        self.df = pd.DataFrame({'alg': pd.Series(dtype='str'),
                                'n': pd.Series(dtype='int'),
                                'bw': pd.Series(dtype='int'),
                                'seed': pd.Series(dtype='int'),
                                'slot_dur_s': pd.Series(dtype='int'),
                                'rho': pd.Series(dtype='int'),
                                'job_id': pd.Series(dtype='int64'),
                                'mice': pd.Series(dtype='int'),
                                'boost': pd.Series(dtype='float'),
                                'num_int': pd.Series(dtype='int'), 
                                'per_int_duration': pd.Series(dtype='float'), 
                                'job_arrival_time': pd.Series(dtype='float'), 
                                'job_completion_time': pd.Series(dtype='float'), 
                                'req_cpu_per_int': pd.Series(dtype='float'), 
                                'req_mem_per_int': pd.Series(dtype='float'), 
                                'req_bw_per_int': pd.Series(dtype='float'), 
                                'av_ef_satsf': pd.Series(dtype='float'), 
                                'av_si_satsf': pd.Series(dtype='float')})

    def append_to_boost(self, job_df, n_u, bwmhz, rho, seed, slot_duration, alg_name_str):
        tmp_boost = Boost_df()
        tmp_boost.df['boost'] = pd.Series(job_df['boost'], dtype='float')
        tmp_boost.df['job_id'] = pd.Series(job_df['job_id'], dtype='int')
        tmp_boost.df['mice'] = pd.Series(job_df['mice'], dtype='int')
        tmp_boost.df['num_int'] = pd.Series(job_df['num_int'], dtype='int')
        tmp_boost.df['per_int_duration'] = pd.Series(job_df['per_int_duration'], dtype='float')
        tmp_boost.df['job_arrival_time'] = pd.Series(job_df['job_arrival_time'], dtype='float')
        tmp_boost.df['job_completion_time'] = pd.Series(job_df['job_completion_time'], dtype='float')
        tmp_boost.df['av_ef_satsf'] = pd.Series(job_df['av_ef_satsf'], dtype='float')
        tmp_boost.df['av_si_satsf'] = pd.Series(job_df['av_si_satsf'], dtype='float')
        tmp_boost.df['req_cpu_per_int'] = pd.Series(job_df['req_cpu_per_int'], dtype='float')
        tmp_boost.df['req_mem_per_int'] = pd.Series(job_df['req_mem_per_int'], dtype='float')
        tmp_boost.df['req_bw_per_int'] = pd.Series(job_df['req_bw_per_int'], dtype='float')
        tmp_boost.df.loc[:, 'n'] = n_u
        tmp_boost.df.loc[:, 'bw'] = bwmhz
        tmp_boost.df.loc[:, 'rho'] = rho
        tmp_boost.df.loc[:, 'seed'] = seed
        tmp_boost.df.loc[:, 'slot_dur_s'] = int(slot_duration)
        tmp_boost.df.loc[:, 'alg'] = alg_name_str
        self.df = self.df.append(tmp_boost.df, ignore_index=True)
        del(tmp_boost)


if __name__ == "__main__":
    # arg 1: seed start (inclusive)
    # arg 2: seed end (inclusive)
    # arg 3: slot duration in second
    seed_start = int(sys.argv[1])
    seed_end = int(sys.argv[2])
    bw_list = [1, 2, 3]
    rho_list = [5, 7, 9]
    nu_list = [10, 15, 20]
    slot_dur_list = [16.0, 64.0, 256.0, 1024.0]
    # bw_list = [1]
    # rho_list = [9]
    # nu_list = [10]
    # slot_dur_list = [64.0]

    boost = Boost_df()
    event_df = pd.DataFrame()
    print(f"starting seed: {seed_start}")
    for slot_duration in slot_dur_list:
        for seed in range(seed_start, seed_end+1):
            for n_u in nu_list:
                for bwmhz in bw_list:
                    for rho in rho_list:
                        input_dict = extr_input_dct(bwmhz, rho, n_u, seed, slot_duration)
                        tsfer_waiting_log = TSFER_waiting_log()
                        tsfer_running_log = Running_log_egal()
                        tsfer_finished_log = Finished_log_egal()
                        tsfer_env = Slotted_TSFER_env(input_dict,
                                                      tsfer_waiting_log,
                                                      tsfer_running_log,
                                                      tsfer_finished_log)
                        tsfer_env.run()
                        init_df, job_df = tsfer_env.save_finished_log()
                        boost.append_to_boost(job_df, n_u, bwmhz, rho, seed, slot_duration, 'tsfer')
                        tsfer_env.event_record.loc[:, 'n'] = n_u
                        tsfer_env.event_record.loc[:, 'bw'] = bwmhz
                        tsfer_env.event_record.loc[:, 'rho'] = rho
                        tsfer_env.event_record.loc[:, 'seed'] = seed
                        tsfer_env.event_record.loc[:, 'slot_dur_s'] = int(slot_duration)
                        tsfer_env.event_record.loc[:, 'alg'] = 'tsfer'
                        event_df = event_df.append(tsfer_env.event_record, ignore_index=True)
                        del(tsfer_env)
                        del(tsfer_waiting_log)
                        del(tsfer_running_log)
                        del(tsfer_finished_log)

                        drfer_waiting_log = DRFER_waiting_log()
                        drfer_running_log = Running_log_egal()
                        drfer_finished_log = Finished_log_egal()
                        drfer_env = Slotted_DRFER_env(input_dict,
                                                      drfer_waiting_log,
                                                      drfer_running_log,
                                                      drfer_finished_log)
                        drfer_env.run()
                        init_df, job_df = drfer_env.save_finished_log()
                        boost.append_to_boost(job_df, n_u, bwmhz, rho, seed, slot_duration, 'drfer')
                        drfer_env.event_record.loc[:, 'n'] = n_u
                        drfer_env.event_record.loc[:, 'bw'] = bwmhz
                        drfer_env.event_record.loc[:, 'rho'] = rho
                        drfer_env.event_record.loc[:, 'seed'] = seed
                        drfer_env.event_record.loc[:, 'slot_dur_s'] = int(slot_duration)
                        drfer_env.event_record.loc[:, 'alg'] = 'drfer'
                        event_df = event_df.append(drfer_env.event_record, ignore_index=True)
                        del(drfer_env)
                        del(drfer_waiting_log)
                        del(drfer_running_log)
                        del(drfer_finished_log)

                        mnw_waiting_log = MNW_waiting_log()
                        mnw_running_log = Running_log_u()
                        mnw_finished_log = Finished_log_u()
                        mnw_env = Slotted_mnw_env(input_dict, 
                                                  mnw_waiting_log,
                                                  mnw_running_log,
                                                  mnw_finished_log)
                        mnw_env.run()
                        init_df, job_df = mnw_env.save_finished_log()
                        boost.append_to_boost(job_df, n_u, bwmhz, rho, seed, slot_duration, 'mnw')
                        mnw_env.event_record.loc[:, 'n'] = n_u
                        mnw_env.event_record.loc[:, 'bw'] = bwmhz
                        mnw_env.event_record.loc[:, 'rho'] = rho
                        mnw_env.event_record.loc[:, 'seed'] = seed
                        mnw_env.event_record.loc[:, 'slot_dur_s'] = int(slot_duration)
                        mnw_env.event_record.loc[:, 'alg'] = 'mnw'
                        event_df = event_df.append(mnw_env.event_record, ignore_index=True)
                        del(mnw_env)
                        del(mnw_waiting_log)
                        del(mnw_running_log)
                        del(mnw_finished_log)

                        cru_waiting_log = Cautious_relative_utilitarian_waiting_log()
                        cru_running_log = Running_log_u()
                        cru_finished_log = Finished_log_u()
                        cru_env = \
                            Slotted_cautious_relative_utilitarian_env(input_dict, 
                                                                      cru_waiting_log,
                                                                      cru_running_log,
                                                                      cru_finished_log)
                        cru_env.run()
                        init_df, job_df = cru_env.save_finished_log()
                        boost.append_to_boost(job_df, n_u, bwmhz, rho, seed, slot_duration, 'cru')
                        cru_env.event_record.loc[:, 'n'] = n_u
                        cru_env.event_record.loc[:, 'bw'] = bwmhz
                        cru_env.event_record.loc[:, 'rho'] = rho
                        cru_env.event_record.loc[:, 'seed'] = seed
                        cru_env.event_record.loc[:, 'slot_dur_s'] = int(slot_duration)
                        cru_env.event_record.loc[:, 'alg'] = 'cru'
                        event_df = event_df.append(cru_env.event_record, ignore_index=True)
                        del(cru_env)
                        del(cru_waiting_log)
                        del(cru_running_log)
                        del(cru_finished_log)
                        print(f"\tfinished slot: {slot_duration}, n_u: {n_u}, bwmhz: {bwmhz}, rho: {rho}")

    j_l = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'google_micro_indivis'))
    boost.df.to_csv(os.path.join(j_l, f"indivis_boost_seed{seed_start}.csv"))
    event_df.to_csv(os.path.join(j_l, f"indivis_event_seed{seed_start}.csv"))
    print(f"finished seed: {seed_start}")
