import pandas as pd
from os import path
import numpy as np
import random as rd
CPU_FREQ = 2.6e9
TASK_EVENTS_CSV_COLNAMES = ['time', 'missing', 'job_id', 'task_idx',
                            'machine_id', 'event_type', 'user',
                            'sched_cls', 'priority', 'cpu_requested',
                            'mem_requested', 'disk', 'restriction']
TASK_EVENTS_CSV_USECOLS = ['time', 'job_id', 'task_idx', 'event_type',
                           'cpu_requested', 'mem_requested']

if __name__ == "__main__":
    google_trace_path = {}
    google_trace_path['root'] = "/scratch/meskarer/clusterdata-2011-2"
    google_trace_path['task_events'] =\
        path.join(google_trace_path['root'], 'task_events')
    google_trace_path['job_total_run_time'] =\
        path.join(google_trace_path['root'], 'job_total_run_time')
    j_loc = path.join(google_trace_path['job_total_run_time'],
                      'job_total_run_time.csv')
    total_run_time_df = pd.read_csv(j_loc)
    keys = total_run_time_df['job_id'].tolist()
    values = total_run_time_df['run_time'].tolist()
    task_run_time_ref_dict = dict(zip(keys, values))

    concat_df = pd.DataFrame()
    for part_number in range(1):
        task_usage_df = pd.read_csv(path.join(google_trace_path['task_events'],
                                              'part-{0:05d}-of-00500.csv.gz'.format(part_number)),
                                    header=None, index_col=False,
                                    compression='gzip',
                                    names=TASK_EVENTS_CSV_COLNAMES,
                                    usecols=TASK_EVENTS_CSV_USECOLS)
        task_usage_df = task_usage_df[task_usage_df.time != 0]
        task_usage_df = task_usage_df[task_usage_df.event_type == 0]
        task_usage_df.drop(['event_type'], axis=1, inplace=True)
        task_usage_df['run_time'] = task_usage_df.job_id.transform(lambda x: task_run_time_ref_dict[x] if x in task_run_time_ref_dict else np.nan)
        task_usage_df.dropna(axis=0, how='any',
                             subset=['time', 'run_time', 'cpu_requested', 'mem_requested'],
                             inplace=True)
        task_usage_df.drop(axis=0,
                           index=task_usage_df[task_usage_df['run_time'] == 0].index,
                           inplace=True)
        task_usage_df.drop(axis=0,
                           index=task_usage_df[task_usage_df['cpu_requested'] == 0].index,
                           inplace=True)
        task_usage_df.drop(axis=0,
                           index=task_usage_df[task_usage_df['mem_requested'] == 0].index,
                           inplace=True)

        task_usage_df.drop(axis=0,
                           index=task_usage_df[task_usage_df['time'] > 600e6 + 1*60*60*1e6].index,
                           inplace=True)
        concat_df = concat_df.append(task_usage_df)
    concat_df = concat_df.groupby(by=['job_id']).agg(time_m = ('time', 'min'),
                                                     num_task = ('task_idx', 'count'),
                                                     cpu_requested = ('cpu_requested', 'mean'),
                                                     mem_requested = ('mem_requested', 'mean'),
                                                     run_time_m = ('run_time', 'mean'))

    found_seed = -1
    seeeed = -1
    while seeeed < 100000 and found_seed < 10000:
        seeeed += 1
        sampled_df = concat_df.sample(n=10, random_state=200+8*seeeed+1, replace=False, axis=0)
        sampled_df.sort_values(by=['time_m'], inplace=True, ascending=True)
        sampled_df.reset_index(drop=False, inplace=True)
        sampled_df.time_m = (sampled_df.time_m - 600e6)/1.0e6
        sampled_df.time_m = sampled_df.time_m - sampled_df.iloc[0].time_m
        sampled_df.time_m = np.round(sampled_df.time_m / 60)
        sampled_df.run_time_m = sampled_df.run_time_m / 60
        np.random.seed(23432)
        sampled_df['bit_rate_requested'] = sampled_df.cpu_requested.transform(lambda x: x * CPU_FREQ / np.random.gamma(4, 200))
        min_bwmhz_for_sim = 2
        is_invalid = any(sampled_df['bit_rate_requested'] / 3.5 / (min_bwmhz_for_sim * 1.0e6) > 1.0)
        if not is_invalid:
            found_seed += 1
            sampled_df.to_csv(f"10job_{found_seed}.csv.gz", header=True, index=True)
            print(sampled_df)
            print(f"Done with sr: {1}, seed: {found_seed}!")
