import pandas as pd
import numpy as np
import random as rd
import pickle as pk
import os
from opt import make_gamma_inv_matrix
SPEC_EFFICIENCY = 3.5


def abstract_env(location2read_and_write, location2read_old_jobs,
                 location2read_old_event, event_reading_dict,
                 server_dict, google_trace_path,
                 task_run_time_ref_dict, run_dict, opt_func, alg_name):
    end_part = event_reading_dict['end_part']
    sample_rate = 1.0
    c, cntd = server_dict['c'], run_dict['cntd']
    save_part, upto_part = run_dict['save_part'], run_dict['upto_part']
    part_number, channel_BW = run_dict['part_number'], run_dict['channel_BW']
    (TimeCurrent, f, resource_requested, pointer, old_job,
     old_requested_resource, event_df, job_log_dct, server_CPU_freq, c_BW) =\
         initializing(cntd, google_trace_path, part_number, sample_rate,
                      task_run_time_ref_dict, location2read_old_jobs,
                      location2read_old_event, event_reading_dict['input_df'],
                      server_dict['bw_MHz'], server_dict['rho'])
    S, R = c.shape
    c_tilde, job = c/np.sum(c, axis=0), np.array([])
    if part_number == end_part:
        check_condition = lambda part_number, job: (part_number < upto_part or
                                                    np.size(job) != 0)
    else:
        check_condition = lambda part_number, job: part_number < upto_part
    while (check_condition(part_number, job)):
        if part_number <= end_part:
            (pointer, arrived_job, arrived_requested_resource, TimeCurrent,
             part_number, f, resource_requested, job_log_dct) =\
                inputjobs_EventDriven(part_number, server_CPU_freq, channel_BW,
                                      pointer, f, resource_requested,
                                      google_trace_path, sample_rate,
                                      task_run_time_ref_dict,
                                      job_log_dct)
            job, requested_resource =\
                jobmerg_EventDriven(old_job, old_requested_resource,
                                    arrived_job, arrived_requested_resource)
            (pointer, part_number, TimeNextArrival, f, resource_requested) =\
                next_arrival_time(part_number, google_trace_path, pointer,
                                  sample_rate, f, resource_requested,
                                  task_run_time_ref_dict)
            # Number of users finished before next arrival event
            # Arrival
            NumfinUser, evnt_type, server_status = 0, 0, 0
            if part_number > end_part:
                TimeNextArrival = np.Infinity
        else:
            arrived_job, arrived_requested_resource = np.array([]), np.array([])
            TimeNextArrival = np.Infinity
            job, requested_resource =\
                jobmerg_EventDriven(old_job, old_requested_resource,
                                    arrived_job, arrived_requested_resource)
        while (True):
            if NumfinUser > 0:
                evnt_type = 1  # Departure
                if np.size(job) == 0:
                    # Server is idle until next arrival
                    server_status = 1 
                    TimeNextEvent = TimeNextArrival
                    event_df = update_event_df(evnt_type, TimeCurrent,
                                               TimeNextEvent, server_status,
                                               [0.0, 0.0, 0.0], [], [],
                                               1.0, 1.0, 0.0,
                                               pointer, [], [], event_df)
                    break
            # Finding next finish time based on the number of allocated tasks.
            d_tilde = requested_resource[:, 0:3] / np.hstack((np.sum(c, axis=0),
                                                              c_BW))
            N = requested_resource.shape[0]
            if N == 1:
                x = np.min(c_tilde / d_tilde[0, 0:R], axis=1, keepdims=True).T
                x *= min(1. / d_tilde[0, R], np.sum(x)) / np.sum(x)
            else:
                x = opt_func(c, c_tilde, d_tilde, N, S, R)
            fin_list = find_next_finish_time(TimeCurrent,
                                             requested_resource[:, 3:4],
                                             np.sum(x, axis=1, keepdims=True))
            TimeNextFinish, fin_idx  = fin_list[0], fin_list[1]
            total_util_list =\
                resource_util(x, d_tilde)
            if TimeNextArrival < TimeNextFinish:
                # There is not enough time to finish any task
                TimeNextEvent = TimeNextArrival
                old_job = job[:, :]
                exe_time = (np.sum(x , axis=1, keepdims=True) *
                            (TimeNextEvent - TimeCurrent))
                time_left = requested_resource[:,3: 4] - exe_time
                old_requested_resource = np.hstack((requested_resource[:, 0:3],
                                                    time_left))
                if alg_name in ['u', 'ru', 'drfer']:
                    av_si_satsf, min_si_satsf, non_si_ratio =\
                        av_min_SI_satf_non_SI_ratio(c_tilde, d_tilde, x, N, R)
                else:
                    av_si_satsf, min_si_satsf, non_si_ratio = 1.0, 1.0, 0.0
                event_df = update_event_df(evnt_type, TimeCurrent,
                                           TimeNextEvent, server_status,
                                           total_util_list, old_job,
                                           old_requested_resource,
                                           av_si_satsf, min_si_satsf,
                                           non_si_ratio, pointer, x,
                                           make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde),
                                           event_df)
                TimeCurrent = TimeNextEvent
                break
            else:
                # At least one of the tasks will be finished
                TimeNextEvent, NumfinUser = TimeNextFinish, NumfinUser + 1
                job_log_dct[job[fin_idx, 0]]['job_compl_dur_m'] +=\
                    (TimeNextFinish - job[fin_idx, 1])
                old_job = np.delete(job, fin_idx, axis=0)
                exe_time = (np.sum(x, axis=1, keepdims=True) *
                            (TimeNextFinish - TimeCurrent))
                old_requested_resource =\
                    np.hstack((np.delete(requested_resource[:, 0:3], fin_idx,
                                         axis=0),
                               np.delete(requested_resource[:, 3:4] - exe_time,
                                         fin_idx, axis=0)))
                if alg_name in ['u', 'ru', 'drfer']:
                    av_si_satsf, min_si_satsf, non_si_ratio =\
                        av_min_SI_satf_non_SI_ratio(c_tilde, d_tilde, x, N, R)
                else:
                    av_si_satsf, min_si_satsf, non_si_ratio = 1.0, 1.0, 0.0
                event_df = update_event_df(evnt_type, TimeCurrent,
                                           TimeNextEvent, server_status,
                                           total_util_list, old_job,
                                           old_requested_resource,
                                           av_si_satsf, min_si_satsf,
                                           non_si_ratio, pointer, x,
                                           make_gamma_inv_matrix(N, S, R, c_tilde, d_tilde),
                                           event_df)
                job, requested_resource = old_job, old_requested_resource
                TimeCurrent = TimeNextEvent
    for this_job_id, this_job_dct in job_log_dct.items():
        this_job_dct['boost'] = this_job_dct['div_sta_jct']/this_job_dct['job_compl_dur_m']
    return event_df, job_log_dct


def create_job_total_run_time_dict(google_trace_path):
    os.makedirs(google_trace_path['job_total_run_time'])
    total_run_time_df = pd.DataFrame()
    task_usage_csv_colnames = ['start_time', 'end_time', 'job_id', 'task_idx',
                               'machine_id', 'CPU_rate', 'canonical_memory_usage',
                               'assigned_memory_usage', 'unmapped_page_cache',
                               'total_page_cache', 'maximum_memory_usage',
                               'disk_IO_time', 'local_disk_space_usage',
                               'maximum CPU rate', 'maximum_disk_IO_time',
                               'cycles_per_instruction',
                               'memory_accesses_per_instruction',
                               'sample_portion', 'aggregation_type',
                               'sampled_CPU_usage']
    task_usage_csv_usecols = ['start_time', 'end_time', 'job_id', 'task_idx']
    for fn in sorted(os.listdir(google_trace_path['task_usage'])):
        print(fn)
        task_usage_df = pd.read_csv(os.path.join(google_trace_path['task_usage'],
                                                 fn), header=None,
                                    index_col=False,
                                    compression='gzip',
                                    names=task_usage_csv_colnames,
                                    usecols=task_usage_csv_usecols)
        task_usage_df['run_time'] = task_usage_df['end_time'] - task_usage_df['start_time']
        task_usage_df.drop(['start_time', 'end_time'], axis=1, inplace=True)
        task_usage_df['run_time'] = task_usage_df['run_time'] / 1e6
        grouped = task_usage_df.groupby(['job_id', 'task_idx'], as_index=False).sum()
        total_run_time_df = pd.concat([grouped, total_run_time_df])
        total_run_time_df = total_run_time_df.groupby(['job_id', 'task_idx'], as_index=False).sum()
    total_run_time_df.drop(['task_idx'], axis=1, inplace=True)
    total_run_time_df = total_run_time_df.groupby('job_id', as_index=False).mean()
    total_run_time_df.to_csv(os.path.join(google_trace_path['job_total_run_time'],
                                          'job_total_run_time.csv'),
                             encoding='utf-8', index=False)
    keys = total_run_time_df['job_id'].tolist()
    values = total_run_time_df['run_time'].tolist()
    task_run_time_ref_dict = dict(zip(keys, values))
    return task_run_time_ref_dict


def av_min_SI_satf_non_SI_ratio(c_tilde, d_tilde, x, N, R):
    gamma = np.zeros((N, 1))
    for user in range(0, N):
        gamma_sum = np.sum(np.min(c_tilde / d_tilde[user:user+1, 0:R], axis=1))
        gamma[user, 0] = min(gamma_sum, 1. / d_tilde[user, R:(R+1)]) / N
        min_si_satsf = 1.0
        non_SI_users = 0.0
    av = 0.0
    for user in range(0, N):
        if (np.sum(x[user, :]) - gamma[user, 0]) < -1e-4:
            non_SI_users += 1
            min_si_satsf = min(min_si_satsf, np.sum(x[user, :]) / gamma[user, 0])
            av += np.sum(x[user, :]) / gamma[user, 0]
        else:
            av += 1

    return av/N, min_si_satsf, non_SI_users/N


def update_event_df(evnt_type, TimeCurrent, TimeNextEvent, server_status,
                    total_util_list, old_job, old_requested_resource,
                    av_SI_satf, min_SI_satf, ratio_non_SI, pointer,
                    x, gamma_inv_mat, event_df):
    if len(old_job) == 0 and len(old_requested_resource) == 0:
        unf = np.array([])
    else:
        unf = np.hstack((old_job, old_requested_resource))
    if x == []:
        N, sum_x, sum_norm_x = 0, 0, 0
        ln = {'event_type': evnt_type, 'current_event_time': TimeCurrent,
              'event_duration': (TimeNextEvent - TimeCurrent),
              'av_SI_satf': av_SI_satf, 'min_SI_satf': min_SI_satf,
              'ratio_non_SI': ratio_non_SI, 'server_status': server_status,
              'total_util': total_util_list, 'unfinished_jobs_in_the_end': unf,
              'next_task_pointer': pointer, 'sum_x': sum_x,
              'sum_norm_x': sum_norm_x, 'N': N}
    else:
        ln = {'event_type': evnt_type, 'current_event_time': TimeCurrent,
              'event_duration': (TimeNextEvent - TimeCurrent),
              'av_SI_satf': av_SI_satf, 'min_SI_satf': min_SI_satf,
              'ratio_non_SI': ratio_non_SI, 'server_status': server_status,
              'total_util': total_util_list, 'unfinished_jobs_in_the_end': unf,
              'next_task_pointer': pointer, 'sum_x': np.sum(x),
              'sum_norm_x': np.sum(np.matmul(gamma_inv_mat, x)), 'N': x.shape[0]}

    event_df = event_df.append(ln, ignore_index=True)
    return event_df


def resource_util(x, d_tilde):
    U_BW = np.sum(np.array(np.sum(x, axis=1)) * np.array(d_tilde[:, 2]))
    tot_U_CPU = np.sum(np.array(np.sum(x, axis=1)) * np.array(d_tilde[:, 0]))
    tot_U_MEM = np.sum(np.array(np.sum(x, axis=1)) * np.array(d_tilde[:, 1]))
    return [100 * tot_U_CPU, 100 * tot_U_MEM, 100 * U_BW]


def find_next_finish_time(TimeCurrent_microsec, remained_task_by_microsec,
                          sum_task_allocated):
    tmp = []
    for i in range(remained_task_by_microsec.shape[0]):
        if sum_task_allocated[i] > 1e-8:
            tmp.append([remained_task_by_microsec[i]/sum_task_allocated[i]])
        else:
            tmp.append([np.inf])
    ExeTime_microsec = np.array(tmp)
    MinExeDur_microsec = np.min(ExeTime_microsec)
    FirstFinUser = np.argmin(ExeTime_microsec)
    return [(TimeCurrent_microsec + MinExeDur_microsec), FirstFinUser]


def next_arrival_time(part_number, google_trace_path, pointer, sample_rate, f,
                      resource_requested, task_run_time_ref_dict):
    if part_number > 1: return (pointer, part_number, np.inf, f, 
                                  resource_requested)
    if np.size(f, 0) >= pointer: return (pointer, part_number,
                                        float(f[pointer - 1, 0]), f,
                                        resource_requested)
    part_number = part_number + 1
    pointer = 1
    return (pointer, part_number, np.inf, f, resource_requested)


def jobmerg_EventDriven(old_job, old_requested_resource, new_job,
                        new_requested_resource):
    if np.size(old_job) == 0: return new_job, new_requested_resource
    # If there are some jobs remained from last time slot
    mrg_requested_resource = np.copy(old_requested_resource)
    mrg_job = np.copy(old_job)
    for i in range(new_job.shape[0]):
        idx = (old_job[:, 0] == new_job[i, 0])
        if np.sum(idx) > 0:
            # If the arrived job is not new
            mrg_requested_resource[idx, 3] =\
                new_requested_resource[i, 3] + mrg_requested_resource[idx, 3]
        else:
            mrg_requested_resource = np.vstack((mrg_requested_resource,
                                                new_requested_resource[i:i+1, :]))
            mrg_job = np.vstack((mrg_job, new_job[i:i+1, :]))
    return mrg_job, mrg_requested_resource


def read_part(google_trace_path, part_number, sample_rate,
              task_run_time_ref_dict, input_df, bw_MHz, rho):
#     task_event_list = sorted(os.listdir(google_trace_path['task_events']))
#     task_usage_df = pd.read_csv(os.path.join(google_trace_path['task_events'],
#                                              task_event_list[part_number-1]),
#                                 header=0, index_col=0,
#                                 compression='gzip')
    env_file_dir = os.path.dirname(__file__)
    google_input_dir = os.path.abspath(os.path.join(env_file_dir, '..', 'google_input'))
#     task_usage_df = pd.read_csv(os.path.join(google_input_dir, f"10job_{google_trace_path['server_seed']}.csv.gz"),
#                                 header=0, index_col=0,
#                                 compression='gzip')
    j = input_df[['time_m','job_id', 'num_task']].to_numpy()
    d = input_df[['cpu_requested', 'mem_requested', f"bw_requested_{bw_MHz}", 'run_time_m',
                  f"div_sta_jct_bw{bw_MHz}_rho{int(rho*10)}",
                  f"div_sta_t_bw{bw_MHz}_rho{int(rho*10)}", 'mice']].to_numpy()
    return j, d


def inputjobs_EventDriven(part_number, CPU_freq, W, pointer, f,
                          resource_requested, google_trace_path, sample_rate,
                          task_run_time_ref_dict, job_log_dct):
    job_counter, TimeCurrent, job_ID = f[pointer - 1, 2], f[pointer - 1, 0], f[pointer - 1, 1]
    required_CPU = resource_requested[pointer - 1, 0]
    required_MEM = resource_requested[pointer - 1, 1]
    # required_bit_rate = resource_requested[pointer - 1, 2]
    # required_BW = required_bit_rate / SPEC_EFFICIENCY / W
    required_BW = resource_requested[pointer - 1, 2]
    execution_duration = resource_requested[pointer - 1, 3]
    div_sta_jct = resource_requested[pointer - 1, 4]
    div_sta_t = resource_requested[pointer - 1, 5]
    mice = resource_requested[pointer - 1, 6]
    arrived_job = np.array([[job_ID, TimeCurrent]])
    arrived_d = np.array([[required_CPU, required_MEM, required_BW, execution_duration * job_counter]])
    if job_ID not in job_log_dct:
        job_log_dct[job_ID] = {'arrival_time_m': TimeCurrent,
                               'per_task_dur_m': execution_duration,
                               'num_task': job_counter,
                               'req_cpu': required_CPU, 'req_mem': required_MEM,
                               'req_bw': required_BW, 'job_compl_dur_m': 0.0,
                               'div_sta_jct': div_sta_jct,
                               'div_sta_t': div_sta_t, 'mice': mice}
    pointer += 1
    while pointer <= np.size(f, 0) and f[pointer - 1, 0] == TimeCurrent:
        job_counter, TimeCurrent, job_ID = f[pointer - 1, 2], f[pointer - 1, 0], f[pointer - 1, 1]
        required_CPU = resource_requested[pointer - 1, 0]
        required_MEM = resource_requested[pointer - 1, 1]
        # required_bit_rate = resource_requested[pointer - 1, 2]
        required_BW = resource_requested[pointer - 1, 2]
        execution_duration = resource_requested[pointer - 1, 3]
        div_sta_jct = resource_requested[pointer - 1, 4]
        div_sta_t = resource_requested[pointer - 1, 5]
        mice = resource_requested[pointer - 1, 6]
        # required_BW = required_bit_rate / SPEC_EFFICIENCY / W
        arrived_job = np.vstack((arrived_job, np.array([[job_ID, TimeCurrent]])))
        arrived_d = np.vstack((arrived_d, np.array([[required_CPU, required_MEM, required_BW, execution_duration * job_counter]])))
        pointer += 1
        if job_ID not in job_log_dct:
            job_log_dct[job_ID] = {'arrival_time_m': TimeCurrent,
                                   'per_task_dur_m': execution_duration,
                                   'num_task': job_counter,
                                   'req_cpu': required_CPU, 'req_mem': required_MEM,
                                   'req_bw': required_BW, 'job_compl_dur_m': 0.0,
                                   'div_sta_jct': div_sta_jct,
                                   'div_sta_t': div_sta_t, 'mice': mice}
    if pointer > np.size(f, 0):
        part_number, pointer = part_number + 1, 1
    return (pointer, arrived_job, arrived_d,
            float(TimeCurrent), part_number,
            f, resource_requested, job_log_dct)


def initializing(cntd, google_trace_path, part_number, sample_rate,
                 task_run_time_ref_dict, location2read_old_jobs,
                 location2read_old_event, input_df, bw_MHz, rho):
    event_df = pd.DataFrame(columns=['current_event_time',
                                     'event_duration', 'event_type',
                                     'server_status', 'total_util',
                                     'next_task_pointer', 'min_SI_satf',
                                     'av_SI_satf', 'ratio_non_SI',
                                     'sum_x', 'sum_norm_x', 'N',
                                     'unfinished_jobs_in_the_end'])
    server_CPU_freq, c_BW = 2.6e9, 1.0
    if cntd != 1:
        # This is the first file that we read
        # and it is not a continuation of any other file or simulation
        job_log_dct = {}
        TimeCurrent = 0
        f, resource_requested = read_part(google_trace_path, part_number,
                                          sample_rate,
                                          task_run_time_ref_dict, input_df,
                                          bw_MHz, rho)
        TimeNextArrival, pointer = f[0, 0], 1
        old_job, old_requested_resource = np.array([]), np.array([])
        event_df =\
            event_df.append({'current_event_time': TimeCurrent,
                             'event_type': -1,
                             'event_duration': TimeNextArrival - TimeCurrent,
                             'server_status': 1, 'min_SI_satf': 1.0, 
                             'av_SI_satf': 1.0, 'ratio_non_SI': 0.0,
                             'total_util': [0.0, 0.0, 0.0],
                             'next_task_pointer': pointer, 'sum_x': 0.0,
                             'sum_norm_x': 0.0, 'N': 0,
                             'unfinished_jobs_in_the_end': np.array([])},
                             ignore_index=True)
    else:
        f, resource_requested = read_part(google_trace_path, part_number,
                                          sample_rate,
                                          task_run_time_ref_dict, input_df,
                                          bw_MHz, rho)
        job_log_dct = pk.load(open(location2read_old_jobs, "rb"))
        last_event = pd.read_pickle(location2read_old_event)
        last_dict_event = last_event.iloc[-1]
        pointer = last_dict_event['next_task_pointer']
        if last_dict_event['server_status'] == 1:
            old_job, old_requested_resource = np.array([]), np.array([])
            TimeCurrent = last_dict_event['current_event_time']\
                + last_dict_event['event_duration']
        else:
            old_job = last_dict_event['unfinished_jobs_in_the_end'][:, 0:2]
            old_requested_resource =\
                last_dict_event['unfinished_jobs_in_the_end'][:, 2:6]
            TimeCurrent = last_dict_event['current_event_time']\
                + last_dict_event['event_duration']
    return (TimeCurrent, f, resource_requested, pointer, old_job,
            old_requested_resource, event_df, job_log_dct, server_CPU_freq, c_BW)


def concat_ev(event_reading_dict, location_dict, alg_name):
    ccat = {'util_cpu': [], 'util_mem': [], 'util_bw': [], 'time_h': [],
            'dur_s': [], 'non_si_ratio': [], 'min_si_satf': [],
            'av_si_satf': [], 'last_arr_h': 0, 'sum_x': [],
            'sum_norm_x': [], 'N': []}
    for part_number in np.arange(event_reading_dict['start_part'], event_reading_dict['end_part']+1):
        file = os.path.join(location_dict['RW_path'], f"event_df_{alg_name}_part_{part_number}.pk")
        data = pd.read_pickle(file)
        for r in range(len(data)):
            ccat['util_cpu'].append(data.iloc[r]['total_util'][0])
            ccat['util_mem'].append(data.iloc[r]['total_util'][1])
            ccat['util_bw'].append(data.iloc[r]['total_util'][2])
            ccat['dur_s'].append(data.iloc[r]['event_duration']/1e6)
            ccat['non_si_ratio'].append(data.iloc[r]['ratio_non_SI'])
            ccat['min_si_satf'].append(data.iloc[r]['min_SI_satf'])
            ccat['av_si_satf'].append(data.iloc[r]['av_SI_satf'])
            ccat['time_h'].append((data.iloc[r]['current_event_time'] - 600e6)/3600e6)
            ccat['sum_x'].append(data.iloc[r]['sum_x'])
            ccat['sum_norm_x'].append(data.iloc[r]['sum_norm_x'])
            ccat['N'].append(data.iloc[r]['N'])
            if data.iloc[r]['event_type'] == 0:
                ccat['last_arr_h'] = (data.iloc[r]['current_event_time'] - 600e6)/3600e6
    ccat['util_cpu'].pop()
    ccat['util_mem'].pop()
    ccat['util_bw'].pop()
    ccat['dur_s'].pop()
    ccat['non_si_ratio'].pop()
    ccat['min_si_satf'].pop()
    ccat['av_si_satf'].pop()
    ccat['sum_x'].pop()
    ccat['sum_norm_x'].pop()
    ccat['N'].pop()
    with open(os.path.join(location_dict['RW_path'], f"concat_event_{alg_name}.pk"), 'wb') as cc_file:
        pk.dump(ccat, cc_file)
