import os
import math
import bisect as bs
import pickle as pk
import pandas as pd
import numpy as np
import sys


class Abs_egal_slotted_env:
    def __init__(self, input_dict, arrival_log, execution_log, departure_log):
        self.waiting_log = arrival_log
        self.running_log = execution_log
        self.finished_log = departure_log
        self.time_current = 0.0
        self.pointer = 1
        self.num_srv = input_dict['num_server']
        self.server_seed = input_dict['server_seed']
        self.channel_BW = input_dict['channel_BW']
        self.bwmhz = input_dict['channel_BW']/1e6
        self.rho = input_dict['rho']
        self.extr_res = input_dict['extr_res_cap']
        self.avail_extr_res = np.copy(self.extr_res) if type(self.extr_res) == np.array else self.extr_res
        self.c = input_dict['comp_res_cap']
        self.avail_comp_res = np.copy(self.c)
        self.samp_rate = input_dict['samp_rate']
        self.slot_duration = input_dict['slot_duration']  # in second
        self.reallocation_overhead = input_dict['realloc_overhead']  # in second
        # setting path parameters
        if os.popen('hostname').read()[0:-1] in ['erfan-Inspiron-3670',
                                                 'DESKTOP-VVQRRIT']:
            self.scratch_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                             '..', '..'))
            if sys.argv[3] == 'a':
                self.trace_path = os.path.join(self.scratch_path, 'alibaba_trace')
            elif sys.argv[3] == 'g':
                self.trace_path = os.path.join(self.scratch_path, 'google_input')
        else:
            self.scratch_path = os.path.join(input_dict['scr_path'])
            if sys.argv[3] == 'a':
                self.trace_path = os.path.join(self.scratch_path, 'alibaba_trace')
            elif sys.argv[3] == 'g':
                self.trace_path = os.path.join(self.scratch_path, 'google_input')
        sim_folder_n = (f"{input_dict['rho']}rho_" +
                        f"{len(input_dict['input_df'])}NumUser" +
                        f"{self.server_seed}randseed" +
                        f"{self.num_srv}NumServer" +
                        f"{int(self.channel_BW / 1e6)}MHzBW" +
                        f"{int(self.slot_duration)}slot_dur")
        self.name_prefix = sim_folder_n
        if sys.argv[3] == 'a':
            self.RW_path = os.path.join(self.scratch_path, 'alibaba_indivis',
                                        sim_folder_n)
            self.RW_path2 = os.path.join(self.scratch_path, 'alibaba_indivis')
        elif sys.argv[3] == 'g':
            self.RW_path = os.path.join(self.scratch_path, 'google_micro_indivis',
                                    sim_folder_n)
            self.RW_path2 = os.path.join(self.scratch_path, 'google_micro_indivis')
        self.event_log_path = os.path.join(self.RW_path2, sim_folder_n + '_event_log.csv')
        self.job_log_path = os.path.join(self.RW_path2, sim_folder_n + '_job_log.csv')
        self.task_log_path = os.path.join(self.RW_path2, sim_folder_n + '_task_log.csv')
        task_event_folder = os.path.join(self.scratch_path, 'alibaba_trace',
                                         'input')
        df_sampled_te = input_dict['input_df']
        df_sampled_te.rename(columns={'time_m': 'job_sub_time',
                                      'num_task': 'int_num',
                                      'cpu_requested': 'req_cpu_core',
                                      'mem_requested': 'req_mem_GB',
                                      'run_time_m': 'int_runtime_sec',
                                      f"bw_requested_{int(self.channel_BW / 1e6)}": 'req_bw',
                                      f"indiv_sta_t_bw{int(self.bwmhz)}_rho{self.rho}": 'sta_t',
                                      f"indiv_sta_jct_bw{int(self.bwmhz)}_rho{self.rho}": 'sta_jct'}, inplace=True)
        self.task_event_df = df_sampled_te.loc[:, ['job_sub_time', 'job_id',
                                                   'int_num', 'req_cpu_core',
                                                   'req_mem_GB',
                                                   'req_bw', 'int_runtime_sec',
                                                   'mice', 'sta_t', 'sta_jct']]
        self.num_task_event_rows = df_sampled_te.shape[0]
        self.time_next_arrival = self.task_event_df.loc[0, 'job_sub_time']
        self.time_next_event = self.task_event_df.loc[0, 'job_sub_time']
        self.time_next_finish = np.inf
        self.time_next_slot = (self.task_event_df.loc[0, 'job_sub_time'] +
                               self.slot_duration)
        self.sth_arrived_or_finished_bw_slot = False
        self.event_record = pd.DataFrame(columns=['current_event_time',
                                                  'avail_comp_res',
                                                  'avail_extr_res',
                                                  'next_task_pointer',
                                                  'Q_length',
                                                  'min_si_satisfaction',
                                                  'av_si_satisfaction',
                                                  'min_ef_satisfaction',
                                                  'av_ef_satisfaction'])
        new_event_ln = {'current_event_time': self.time_current,
                        'avail_comp_res': self.avail_comp_res.tolist(),
                        'avail_extr_res': [self.avail_extr_res] if np.shape(self.avail_extr_res) == () else self.avail_extr_res.tolist(),
                        'next_task_pointer': self.pointer, 'Q_length': 0,
                        'min_si_satisfaction': 1.0, 'av_si_satisfaction': 1.0,
                        'min_ef_satisfaction': 1.0, 'av_ef_satisfaction': 1.0}
        self.event_record = self.event_record.append(new_event_ln,
                                                     ignore_index=True)

    def save_finished_log(self):
        df_int_log = pd.DataFrame(columns=['job_id', 'int_id',
                                           'int_arrival_time', 'int_wait_time',
                                           'int_completion_time'])
        df_job_log = pd.DataFrame(columns=['job_id', 'num_int',
                                           'per_int_duration',
                                           'job_arrival_time',
                                           'job_completion_time',
                                           'req_cpu_per_int', 'req_mem_per_int',
                                           'req_bw_per_int', 'mice',
                                           'av_ef_satsf', 'av_si_satsf',
                                           'boost'])
        for job_id, job in self.finished_log.finished_job_dct.items():
            num_task = len(job.finished_task_dct)
            per_int_duration = job.one_task_duration
            job_arrival_time = job.job_arrival_time
            req_cpu_per_int = job.demanded_res[0]
            req_mem_per_int = job.demanded_res[1]
            req_bw_per_int = job.demanded_res[2]
            mice = job.mice
            sta_jct = job.sta_jct
            sum_ef_satsf = job.sum_min_ef_satsf
            sum_si_satsf = job.sum_si_satsf
            job_completion_time = 0.0
            for task_id, task in job.finished_task_dct.items():
                int_arrival_time = task.task_arrival_time
                int_wait_time = task.waited_time
                int_completion_time = task.finish_time
                df_int_log = df_int_log.append({'job_id': job_id, 
                                                'int_id': task_id,
                                                'int_arrival_time': int_arrival_time,
                                                'int_wait_time': int_wait_time,
                                                'int_completion_time': int_completion_time},
                                                ignore_index=True)
                job_completion_time = max(job_completion_time, int_completion_time)
            av_ef_satsf = sum_ef_satsf/(job_completion_time - job_arrival_time)
            av_si_satsf = sum_si_satsf/(job_completion_time - job_arrival_time)
            df_job_log = df_job_log.append({'job_id': job_id, 'num_int': num_task,
                                            'per_int_duration': per_int_duration,
                                            'job_arrival_time': job_arrival_time,
                                            'job_completion_time': job_completion_time,
                                            'req_cpu_per_int': req_cpu_per_int,
                                            'req_mem_per_int': req_mem_per_int,
                                            'req_bw_per_int': req_bw_per_int,
                                            'mice': mice,
                                            'av_ef_satsf': av_ef_satsf,
                                            'av_si_satsf': av_si_satsf,
                                            'boost': sta_jct/(job_completion_time - job_arrival_time)},
                                            ignore_index=True)
        # df_job_log.to_csv(self.job_log_path, header=True, index=True)
        # df_int_log.to_csv(self.task_log_path, header=True, index=True)
        return df_int_log, df_job_log

    def load_finished_log(self):
        pk.dump(self.finished_log, open(self.queue_dict_path, "wb"))
    
    def save_df_event(self):
        self.event_record.to_csv(self.event_log_path, header=True, index=True)

    def sim_not_over(self):
        return (self.time_next_arrival != np.inf or 
                self.waiting_log.waiting_job_dct or
                self.running_log.j_t_idx_sorted_by_tent_fin_time)
    
    def get_min_av_ef_satisfaction(self):
        job_set = set().union(self.running_log.running_job_dct.keys(),
                              self.waiting_log.waiting_job_dct.keys())
        N, (S, R) = len(job_set), self.c.shape
        if N < 1: return 1.0, 1.0
        sum_ef_satsf, min_ef_satsf = 0.0, 1
        for job_id in job_set:
            if job_id not in self.running_log.running_job_dct:
                job = self.waiting_log.waiting_job_dct[job_id]
            else:
                job = self.running_log.running_job_dct[job_id]
            num_alloc_task = len(job.running_task_dct)
            demand_res = job.demanded_res
            total_num_task = len(job.running_task_dct) + len(job.waiting_task_dct)
            # Find the number of tasks it get from another users' alloc
            min_ef_satsf_for_this_user = 1
            for job_id_2 in job_set:
                if job_id_2 == job_id:
                    continue
                num_task_with_2 = 0
                if job_id_2 not in self.running_log.running_job_dct:
                    num_task_with_2 = 0
                else:
                    job2 = self.running_log.running_job_dct[job_id_2]
                    demand_res2 = job2.demanded_res
                    num_task_in_s = {s: 0 for s in range(S)}
                    if job2.running_task_dct:
                        for task in job2.running_task_dct.values():
                            num_task_in_s[task.allocated_server] += 1
                    if np.sum([x for x in num_task_in_s.values()]) > 0:
                        for s in range(0, S):
                            num_task_with_2 += math.floor(np.min(num_task_in_s[s] * demand_res2[0: R]/demand_res[0: R]))
                        if math.floor(np.sum([x for x in num_task_in_s.values()]) * demand_res2[R:]/demand_res[R:]) <= num_task_with_2:
                            num_task_with_2 = math.floor(np.sum([x for x in num_task_in_s.values()]) * demand_res2[R:]/demand_res[R:])
                    else:
                        num_task_with_2 = 0
                    num_task_with_2 = min(num_task_with_2, total_num_task, job.sta_t)
                ef_satsf = 1 if num_task_with_2 == 0 else min(num_alloc_task/num_task_with_2, 1)
                min_ef_satsf_for_this_user = min(min_ef_satsf_for_this_user, ef_satsf)
                min_ef_satsf = min(min_ef_satsf, ef_satsf)
                sum_ef_satsf += ef_satsf
            job.sum_min_ef_satsf += min_ef_satsf_for_this_user * (self.time_next_event - self.time_current)
        if N < 2:
            return 1.0, 1.0
        else:
            return min_ef_satsf, sum_ef_satsf/(N*(N-1))
    
    def get_min_av_si_satisfaction(self):
        job_set = set().union(self.running_log.running_job_dct.keys(),
                              self.waiting_log.waiting_job_dct.keys())
        N, (S, R) = len(job_set), self.c.shape
        if N == 0: return 1.0, 1.0
        sum_si_satsf, min_si_satsf = 0.0, 1.0
        for job_id in job_set:
            if job_id not in self.running_log.running_job_dct:
                si_satsf = 0
                job = self.waiting_log.waiting_job_dct[job_id]
            else:
                job = self.running_log.running_job_dct[job_id]
                si_task = 0.0
                demand_res = job.demanded_res
                for s in range(0, S):
                    si_task += math.floor(np.min(self.c[s, 0:R]/N/demand_res[0:R]))
                ext = self.extr_res if np.shape(self.extr_res) == () else np.sum(self.extr_res)
                if math.floor(np.min(ext/N/demand_res[R:])) <= si_task:
                    si_task = math.floor(np.min(ext/N/demand_res[R:]))
                total_task = len(job.running_task_dct) + len(job.waiting_task_dct)
                si_task = min(total_task, si_task)
                running_task = len(job.running_task_dct)
                si_satsf = 1.0 if si_task == 0 else min(running_task/si_task, 1.0)
            job.sum_si_satsf += si_satsf * (self.time_next_event - self.time_current)
            min_si_satsf = min(min_si_satsf, si_satsf)
            sum_si_satsf += si_satsf
        av_si_satsf = sum_si_satsf/N
        return min_si_satsf, av_si_satsf

    def update_df_event(self):
        min_si_satsf, av_si_satsf = self.get_min_av_si_satisfaction()
        min_ef_satsf, av_ef_satsf = self.get_min_av_ef_satisfaction()
        nln = {'current_event_time': self.time_current,
               'avail_comp_res': self.avail_comp_res.tolist(),
               'avail_extr_res': [self.avail_extr_res] if np.shape(self.avail_extr_res) == () else self.avail_extr_res.tolist(),
               'next_task_pointer': self.pointer,
               'Q_length': self.waiting_log.get_Q_length(),
               'min_si_satisfaction': min_si_satsf,
               'av_si_satisfaction': av_si_satsf,
               'min_ef_satisfaction': min_ef_satsf,
               'av_ef_satisfaction': av_ef_satsf}
        self.event_record = self.event_record.append(nln, ignore_index=True)

    def arrival_event(self):
        return (self.time_next_arrival <= self.time_next_finish and
                self.time_next_arrival <= self.time_next_slot and
                self.time_next_arrival != np.inf)

    def departure_event(self):
        return (self.time_next_finish <= self.time_next_slot and
                  self.time_next_finish <= self.time_next_arrival and
                  self.time_next_finish != np.inf)

    def reallocation_event(self):
        return (self.time_next_slot < self.time_next_arrival and
                  self.time_next_slot < self.time_next_finish)
    
    def read_from_task_event_df(self):
        self.time_current = self.task_event_df.loc[self.pointer - 1,
                                                   'job_sub_time']
        arrived_lst = []
        while (self.pointer <= self.num_task_event_rows and
               self.task_event_df.\
                   loc[self.pointer -1, 'job_sub_time'] == self.time_current):
            job_ID = self.task_event_df.loc[self.pointer - 1, 'job_id']
            num_inst = self.task_event_df.loc[self.pointer - 1, 'int_num']
            single_task_execution_duration = self.task_event_df.\
                                                loc[self.pointer - 1,
                                                    'int_runtime_sec']
            required_CPU = self.task_event_df.loc[self.pointer - 1, 'req_cpu_core']
            required_MEM = self.task_event_df.loc[self.pointer - 1, 'req_mem_GB']
            required_BW = self.task_event_df.loc[self.pointer - 1, 'req_bw']
            mice = self.task_event_df.loc[self.pointer - 1, 'mice']
            sta_t = self.task_event_df.loc[self.pointer - 1, 'sta_t']
            sta_jct = self.task_event_df.loc[self.pointer - 1, 'sta_jct']
            arrived_lst += [[job_ID, inst_id, self.time_current, required_CPU,
                             required_MEM, required_BW,
                             single_task_execution_duration,
                             mice, sta_t, sta_jct] for inst_id in range(int(num_inst))]
            self.pointer += 1
        arrived_job = pd.DataFrame(data=arrived_lst, columns=['job_id',
                                                              'int_id',
                                                              'int_sub_time',
                                                              'req_cpu_core',
                                                              'req_mem_GB',
                                                              'req_bw',
                                                              'int_runtime_sec',
                                                              'mice',
                                                              'sta_t',
                                                              'sta_jct'])
        self.waiting_log.create_and_add_jobs_to_wait_job_dct(arrived_job, self.c,
                                                             self.extr_res)

    def increase_avail_res(self, departing_job_id, departing_task_idx):
        job = self.running_log.running_job_dct[departing_job_id]
        s = job.running_task_dct[departing_task_idx].allocated_server
        if np.shape(self.avail_extr_res) == ():
            self.avail_comp_res[s, :] = (self.avail_comp_res[s, :] +
                                         job.demanded_res[0:-1])
            self.avail_extr_res = self.avail_extr_res + job.demanded_res[-1]
        else:
            self.avail_comp_res[s, :] = (self.avail_comp_res[s, :] +
                                         job.demanded_res[0:-1])
            self.avail_extr_res[s, 0] = (self.avail_extr_res[s, 0] +
                                         job.demanded_res[-1])
    
    def reduce_avail_res(self):
        if np.shape(self.avail_extr_res) == ():
            self.avail_comp_res[self.waiting_log.picked_server, :] =\
                (self.avail_comp_res[self.waiting_log.picked_server, :] -
                 self.waiting_log.\
                     waiting_job_dct[self.waiting_log.picked_job_id].\
                         demanded_res[0:-1])
            self.avail_extr_res = \
                (self.avail_extr_res -
                 self.waiting_log.\
                    waiting_job_dct[self.waiting_log.picked_job_id].\
                        demanded_res[-1])
        else:
            self.avail_comp_res[self.waiting_log.picked_server, :] =\
                (self.avail_comp_res[self.waiting_log.picked_server, :] -
                 self.waiting_log.\
                     waiting_job_dct[self.waiting_log.picked_job_id].\
                         demanded_res[0:-1])
            self.avail_extr_res[self.waiting_log.picked_server, 0] =\
                (self.avail_extr_res[self.waiting_log.picked_server, 0] -
                 self.waiting_log.\
                     waiting_job_dct[self.waiting_log.picked_job_id].\
                         demanded_res[-1])

    def allocate_from_waiting_job_dct(self, best_fit=True):
        self.waiting_log.check_Q(self.avail_comp_res, self.avail_extr_res,
                                  self.running_log.running_job_dct,
                                  best_fit=best_fit)
        while self.waiting_log.can_allocate:
            #Remove from Q and add to job_dict                  
            if (self.waiting_log.picked_job_id in self.running_log.running_job_dct and
                (not self.waiting_log.it_is_a_resubmitted_task)):
                self.reduce_avail_res()
                self.running_log.\
                    update_allocated_task_and_tent_fin(self.waiting_log,
                                                       self.time_current)
            elif (self.waiting_log.picked_job_id not in self.running_log.running_job_dct):
                self.running_log.\
                    running_job_dct[self.waiting_log.picked_job_id] =\
                    self.waiting_log.\
                        waiting_job_dct[self.waiting_log.picked_job_id]
                self.reduce_avail_res()
                self.running_log.\
                    update_allocated_task_and_tent_fin(self.waiting_log,
                                                       self.time_current)
            # Remove picked_job_id from env.waiting_log.job_dict if necessary
            # Remove env.waiting_log.picked_task_idx from job.waiting_task_dct
            if self.waiting_log.picked_job_over:
                self.waiting_log.\
                    waiting_job_dct.pop(self.waiting_log.picked_job_id)
            self.waiting_log.check_Q(self.avail_comp_res, self.avail_extr_res,
                                     self.running_log.running_job_dct,
                                     best_fit=True)
        
    def run(self):
        while self.sim_not_over():
            if self.arrival_event():
                # read from df and create and
                # add jobs (and their tasks) to env.waiting_log.waiting_job_dct
                self.read_from_task_event_df()
                self.sth_arrived_or_finished_bw_slot = True
                # Allocate from env.waiting_log.waiting_job_dct if possible
                self.allocate_from_waiting_job_dct()
                # update times
                self.time_next_finish =\
                    self.running_log.j_t_idx_sorted_by_tent_fin_time[0][0] if self.running_log.j_t_idx_sorted_by_tent_fin_time else np.Infinity
                self.time_next_arrival = \
                    float(self.task_event_df.loc[self.pointer - 1, 'job_sub_time']) if self.task_event_df.shape[0] >= self.pointer else np.inf
                self.time_next_event = min(self.time_next_arrival,
                                           self.time_next_finish,
                                           self.time_next_slot)
                self.update_df_event()
                self.time_current = self.time_next_event
            elif self.departure_event():
                temp = self.running_log.j_t_idx_sorted_by_tent_fin_time.pop(0)
                self.time_current = temp[0]
                departing_j_t_list = temp[1]
                # num_departing_task = len(departing_j_t_list)
                for [departing_job_id, departing_task_idx] in departing_j_t_list:
                    self.increase_avail_res(departing_job_id, departing_task_idx)
                    job = self.running_log.running_job_dct[departing_job_id]
                    task = job.running_task_dct.pop(departing_task_idx)
                    task.finish_time = self.time_current
                    task.executed_time = job.one_task_duration
                    job.finished_task_dct[departing_task_idx] = task
                    self.finished_log.finished_job_dct[departing_job_id] = job
                    job.active_fair_share -= job.one_task_fair_share
                    if job.active_fair_share < 1e-10:
                        job.active_fair_share = 0.0
                        self.running_log.running_job_dct.pop(departing_job_id)
                self.sth_arrived_or_finished_bw_slot = True
                # Allocate from env.waiting_log.waiting_job_dct if possible
                self.allocate_from_waiting_job_dct()
                # update times
                self.time_next_finish =\
                    self.running_log.j_t_idx_sorted_by_tent_fin_time[0][0] if self.running_log.j_t_idx_sorted_by_tent_fin_time else np.Infinity
                self.time_next_event = min(self.time_next_arrival,
                                           self.time_next_finish,
                                           self.time_next_slot)
                self.update_df_event()
                self.time_current = self.time_next_event
            elif self.reallocation_event():
                if not self.sth_arrived_or_finished_bw_slot:
                    if (self.time_next_arrival == np.inf and
                        self.time_next_finish == np.inf):
                        print(f"WARNING. UNALLOCATABLE JOB IN THE SYSTEM")
                        print(self.waiting_log.waiting_job_dct.keys())
                        print(f"{[self.waiting_log.waiting_job_dct[k].demanded_res for k in self.waiting_log.waiting_job_dct]}")
                        break
                    k1 = (min(self.time_next_arrival, self.time_next_finish) - self.time_current) // self.slot_duration
                    self.time_next_slot += (k1 + 1) * self.slot_duration
                    self.time_next_event = min(self.time_next_arrival, self.time_next_finish, self.time_next_slot)
                    self.update_df_event()
                    self.time_current = self.time_next_event
                else:
                    # stop any running task. apply the over_head. Zero the jobs active_fair_share
                    # add (if necessary) all jobs to env.waiting_log.waiting_job_dct
                    # and move all tasks to job.waiting_task_dct
                    # remove all jobs from env.running_log.running_jobs_dct
                    # allocate from waiting jobs and update time
                    for job_id, job in self.running_log.running_job_dct.items():
                        if job_id not in self.waiting_log.waiting_job_dct:
                            self.waiting_log.waiting_job_dct[job_id] = job
                        job.active_fair_share = 0.0
                        for task_id, task in job.running_task_dct.items():
                            task.last_time_stamp_added_to_waiting_task_dct = self.time_next_slot
                            task.executed_time += max(self.time_next_slot - task.last_time_stamp_added_to_running_task_dct - self.reallocation_overhead, 0.0)
                            task.allocated_server = -1
                            task.finish_time = -1.0
                            job.waiting_task_dct[task_id] = task
                            # update job.task_idx_sorted_by_executed_time
                            i = bs.bisect_left([x[0] for x in job.task_idx_sorted_by_executed_time], task.executed_time)
                            if i == len(job.task_idx_sorted_by_executed_time):
                                job.task_idx_sorted_by_executed_time.append([task.executed_time, task_id])
                            else:
                                job.task_idx_sorted_by_executed_time[i:i] = [[task.executed_time, task_id]]
                        job.running_task_dct = {}
                    self.running_log.running_job_dct = {}
                    self.running_log.j_t_idx_sorted_by_tent_fin_time = []
                    self.avail_comp_res = np.copy(self.c)
                    self.avail_extr_res = self.extr_res if np.shape(self.extr_res) == () else np.copy(self.extr_res)
                    self.allocate_from_waiting_job_dct()
                    self.time_next_finish = self.running_log.j_t_idx_sorted_by_tent_fin_time[0][0] if self.running_log.j_t_idx_sorted_by_tent_fin_time else np.Infinity
                    self.time_next_slot += self.slot_duration
                    self.time_next_event = min(self.time_next_arrival, self.time_next_finish, self.time_next_slot)
                    self.update_df_event()
                    self.sth_arrived_or_finished_bw_slot = False
                    self.time_current = self.time_next_event


class Slotted_TSFER_env(Abs_egal_slotted_env):
    def __init__(self, input_dict, arrival_tracker, execution_tracker, departure_tracker):
        super().__init__(input_dict, arrival_tracker, execution_tracker, departure_tracker)
        self.event_log_path = os.path.join(self.RW_path2, self.name_prefix + '_event_log_tsfer.csv')
        self.job_log_path = os.path.join(self.RW_path2, self.name_prefix + '_job_log_tsfer.csv')
        self.task_log_path = os.path.join(self.RW_path2, self.name_prefix + '_task_log_tsfer.csv')


class Slotted_DRFER_env(Abs_egal_slotted_env):
    def __init__(self, input_dict, arrival_tracker, execution_tracker, departure_tracker):
        super().__init__(input_dict, arrival_tracker, execution_tracker, departure_tracker)
        self.event_log_path = os.path.join(self.RW_path2, self.name_prefix + '_event_log_drfer.csv')
        self.job_log_path = os.path.join(self.RW_path2, self.name_prefix + '_job_log_drfer.csv')
        self.task_log_path = os.path.join(self.RW_path2, self.name_prefix + '_task_log_drfer.csv')


class Slotted_Naive_TSF_env(Abs_egal_slotted_env):
    def __init__(self, input_dict, arrival_tracker, execution_tracker, departure_tracker):
        super().__init__(input_dict, arrival_tracker, execution_tracker, departure_tracker)
        self.event_log_path = os.path.join(self.RW_path2, self.name_prefix + '_event_log_naive_tsf.csv')
        self.job_log_path = os.path.join(self.RW_path2, self.name_prefix + '_job_log_naive_tsf.csv')
        self.task_log_path = os.path.join(self.RW_path2, self.name_prefix + '_task_log_naive_tsf.csv')


class Slotted_Naive_DRF_env(Abs_egal_slotted_env):
    def __init__(self, input_dict, arrival_tracker, execution_tracker, departure_tracker):
        super().__init__(input_dict, arrival_tracker, execution_tracker, departure_tracker)
        self.event_log_path = os.path.join(self.RW_path2, self.name_prefix + '_event_log_naive_drf.csv')
        self.job_log_path = os.path.join(self.RW_path2, self.name_prefix + '_job_log_naive_drf.csv')
        self.task_log_path = os.path.join(self.RW_path2, self.name_prefix + '_task_log_naive_drf.csv')
