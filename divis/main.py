from read_input import read_argv
from env import create_job_total_run_time_dict, abstract_env, concat_ev
import opt
import pandas as pd
from os import path
import numpy as np
import pickle as pk
import sys
import helper
from datetime import datetime


class Boost_df:
    def __init__(self):
        self.df = pd.DataFrame({'job_id': pd.Series(dtype='int64'),
                                'n': pd.Series(dtype='int'),
                                'bw': pd.Series(dtype='int'),
                                'seed': pd.Series(dtype='int'),
                                'rho': pd.Series(dtype='int'),
                                'mice': pd.Series(dtype='int'),
                                'boost': pd.Series(dtype='float')})

    def add_job(self, this_job_id, this_job_dct, N, bw_MHz, sampling_seed, rho_int):
        tmp_df = {'job_id': this_job_id,
                  'n': N,
                  'bw': bw_MHz,
                  'seed': sampling_seed,
                  'rho': rho_int,
                  'mice': this_job_dct['mice'],
                  'boost': this_job_dct['boost']}
        self.df = self.df.append(tmp_df, ignore_index=True)

    def add_job_dct(self, input_job_dct, N, bw_MHz, sampling_seed, rho_int):
        for this_job_id, this_job_dct in input_job_dct.items():
            self.add_job(this_job_id, this_job_dct, N, bw_MHz, sampling_seed, rho_int)


if __name__ == "__main__":
    seed_start = int(sys.argv[1]) # inclusive
    seed_end = int(sys.argv[2]) # inclusive
    bw_list = [1, 2, 3, 4]
    rho_list = [5, 7, 9]
    nu_list = [10, 15, 20]
    opt_dct = {'tsfer': opt.tsfer, 'drfer': opt.drfer, 'mnw': opt.mnw, 'cru': opt.cru}
    boost_df_dct = {'tsfer':Boost_df(), 'drfer':Boost_df(), 'mnw':Boost_df(), 'cru':Boost_df()}
    for seed in range(seed_start, seed_end+1):
        for N in nu_list:
            for bwmhz in bw_list:
                for rho in rho_list:
                    location_dict, event_reading_dict, server_dict, channel_BW = helper.read_argv(bwmhz, rho, N, seed)
                    google_trace_path, scratch, run_dict = {}, {}, {'channel_BW': channel_BW}
                    google_trace_path['root'] = location_dict['trace_path']
                    scratch['root'] = location_dict['scratch_path']
                    google_trace_path['job_events'] =\
                        path.join(google_trace_path['root'], 'job_events')
                    google_trace_path['machine_attributes'] =\
                        path.join(google_trace_path['root'], 'machine_attributes')
                    google_trace_path['machine_events'] =\
                        path.join(google_trace_path['root'], 'machine_events')
                    google_trace_path['task_constraints'] =\
                        path.join(google_trace_path['root'], 'task_constraints')
                    google_trace_path['task_usage'] =\
                        path.join(google_trace_path['root'], 'task_usage')
                    google_trace_path['job_total_run_time'] =\
                        path.join(google_trace_path['root'], 'job_total_run_time')
                    c = server_dict['c']
                    part_number = 1
                    run_dict['cntd'] = 0
                    run_dict['save_part'] = 'part_1'
                    run_dict['upto_part'] = 1 + event_reading_dict['jump']
                    run_dict['part_number'] = 1
                    event_df_dct, job_log_dct = {}, {}
                    for alg_name in ['tsfer', 'drfer', 'mnw', 'cru']:
                        event_df_dct[alg_name], job_log_dct[alg_name] =\
                            abstract_env('', '', '', event_reading_dict, server_dict, google_trace_path,
                                         {}, run_dict, opt_dct[alg_name], alg_name)
                        boost_df_dct[alg_name].add_job_dct(job_log_dct[alg_name], N, bwmhz, seed, rho)
    j_l = path.abspath(path.join(path.dirname(__file__), '..', 'google_micro_divis'))
    for alg_name in ['tsfer', 'drfer', 'mnw', 'cru']:
        file_name = f"divis_{alg_name}_boost_seed{seed_start}_{seed_end}.csv"
        boost_df_dct[alg_name].df.to_csv(path.join(j_l, file_name))
