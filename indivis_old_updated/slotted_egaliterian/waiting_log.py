import numpy as np
from random import choice
from .job import Task, Job


class Waiting_log:
    def __init__(self, fair_share_func, is_feasible_on_mach, find_best_fit_server):
        self.waiting_job_dct = {}  # {str(job_id): Abs_sub_job}
        self.can_allocate = False
        self.picked_job_id = ''
        self.picked_server = 0
        self.picked_task_idx = 0
        self.picked_task_arrival_time = 0.0
        self.picked_task_executed_time = 0.0
        self.picked_job_over = False
        self.it_is_a_resubmitted_task = False
        self.find_fair_share = fair_share_func
        self.is_feasible_on_mach = is_feasible_on_mach
        self.find_best_fit_server = find_best_fit_server

    def create_and_add_jobs_to_wait_job_dct(self, arrived_job,
                                            serv_cap, extr_cap):
        num_arrived_task = arrived_job.shape[0]
        temp_pointer = 0
        while temp_pointer < num_arrived_task:
            arr_job_id = arrived_job.loc[temp_pointer, 'job_id']
            if arr_job_id not in self.waiting_job_dct:
                demand_vec = arrived_job.loc[temp_pointer, ['req_cpu_core', 'req_mem_GB', 'req_bw']].to_numpy()
                self.waiting_job_dct[arr_job_id] =\
                    Job(self.find_fair_share(demand_vec, serv_cap, extr_cap), 
                        arrived_job.loc[temp_pointer, 'int_runtime_sec'],
                        arrived_job.loc[temp_pointer, 'int_sub_time'],
                        demand_vec,
                        arrived_job.loc[temp_pointer, 'mice'],
                        arrived_job.loc[temp_pointer, 'sta_t'],
                        arrived_job.loc[temp_pointer, 'sta_jct'])
            while (temp_pointer < num_arrived_task and
                   arrived_job.loc[temp_pointer, 'job_id'] == arr_job_id):
                int_id = arrived_job.loc[temp_pointer, 'int_id']
                int_sub_time = arrived_job.loc[temp_pointer, 'int_sub_time']
                self.waiting_job_dct[arr_job_id].waiting_task_dct[int_id] =\
                    Task(int_sub_time)
                # update self.waiting_job_dct[arr_job_id].task_idx_sorted_by_executed_time
                # any task added at this point has task.executed_time = 0.
                # hence we simply append each task to the end of the list
                self.waiting_job_dct[arr_job_id].\
                    task_idx_sorted_by_executed_time.\
                        append([self.waiting_job_dct[arr_job_id].waiting_task_dct[int_id].executed_time,
                                int_id])
                temp_pointer += 1

    def get_jobs_with_min_fair_share(self, avail_comp, avail_extr):
        min_share, min_job = np.Infinity, []
        S, R = avail_comp.shape
        for job_id, job in self.waiting_job_dct.items():
            found_a_server, s = False, 0
            while (not found_a_server) and (s < S):
                if self.is_feasible_on_mach(s, avail_comp, avail_extr, job.demanded_res):
                    found_a_server = True
                else:
                    s += 1
            if not found_a_server:
                continue
            if job.active_fair_share < min_share:
                min_share = job.active_fair_share
                min_job = [job_id]
            elif job.active_fair_share == min_share:
                min_job.append(job_id)
        return min_job, min_share

    def get_jobs_resulting_updated_min_fair_share(self, min_job, min_share, avail_comp, avail_extr):
        S, R = avail_comp.shape
        best_min_job, best_possible_share = [], np.Infinity
        if not min_job:
            return []
        for job_id in min_job:
            job = self.waiting_job_dct[job_id]
            found_a_server, s = False, 0
            while (not found_a_server) and (s < S):
                if self.is_feasible_on_mach(s, avail_comp, avail_extr, job.demanded_res):
                    found_a_server = True
                else:
                    s += 1
            if found_a_server:
                if min_share + job.one_task_fair_share < best_possible_share:
                    best_possible_share = min_share + job.one_task_fair_share
                    best_min_job = [job_id]
                elif min_share + job.one_task_fair_share == best_possible_share:
                    best_min_job.append(job_id)
        return best_min_job
    
    def pick_from_best_min_job(self, best_min_job, avail_comp, avail_extr, best_fit=True):
        S, _ = avail_comp.shape
        if not best_min_job:
            can_allocate, picked_job_id, picked_server, picked_task_idx, picked_task_arrival_time, picked_task_executed_time = False, [], [], [], [], []
        elif not best_fit:
            can_allocate = True
            picked_job_id = choice(best_min_job)
            job = self.job_dict[picked_job_id]
            picked_server = 0
            for s in list(np.random.permutation(range(0, S))):
                if self.is_feasible_on_mach(s, avail_comp, avail_extr, job.demanded_res):
                    picked_server = s
                    break
            temp = job.task_idx_sorted_by_executed_time.pop(-1)
            picked_task_executed_time = temp[0]
            picked_task_idx = temp[1]
            picked_task_arrival_time = job.waiting_task_dct[picked_task_idx].task_arrival_time
        else:
            can_allocate = True
            picked_job_id = choice(best_min_job)
            job = self.waiting_job_dct[picked_job_id]
            picked_server = self.find_best_fit_server(avail_comp, avail_extr, job.demanded_res)
            temp = job.task_idx_sorted_by_executed_time.pop(-1)
            picked_task_executed_time = temp[0]
            picked_task_idx = temp[1]
            picked_task_arrival_time = job.waiting_task_dct[picked_task_idx].task_arrival_time
        self.can_allocate = can_allocate
        self.picked_job_id = picked_job_id
        self.picked_server = picked_server
        self.picked_task_idx = picked_task_idx
        self.picked_task_arrival_time = picked_task_arrival_time
        self.picked_task_executed_time = picked_task_executed_time

    def check_Q(self, avail_comp, avail_extr, runnning_job_dct, best_fit=True):
        min_job, min_share = self.get_jobs_with_min_fair_share(avail_comp, avail_extr)
        best_min_job = self.get_jobs_resulting_updated_min_fair_share(min_job, min_share, avail_comp, avail_extr)
        self.pick_from_best_min_job(best_min_job, avail_comp, avail_extr, best_fit=best_fit)
        if self.can_allocate:
            job = self.waiting_job_dct[self.picked_job_id]
            self.picked_job_over = not job.task_idx_sorted_by_executed_time
            if self.picked_job_id not in runnning_job_dct:
                self.it_is_a_resubmitted_task = False
            else:
                task_already_active = self.picked_task_idx in runnning_job_dct[self.picked_job_id].running_task_dct
                task_already_finished = self.picked_task_idx in runnning_job_dct[self.picked_job_id].finished_task_dct
                self.it_is_a_resubmitted_task = task_already_active or task_already_finished

    def get_Q_length(self):
        queueLength = 0
        for job in self.waiting_job_dct.values():
            queueLength += len(job.waiting_task_dct)
        return queueLength


class TSFER_waiting_log(Waiting_log):
    def __init__(self):
        def find_one_task_tsfer_share(demand_res, comp_cap, extr_cap):
            (S, R), temp = comp_cap.shape, 0
            for s in range(0, S):
                temp += np.min(comp_cap[s, 0:R]/demand_res[0: R])
            return extr_cap/temp if temp <= extr_cap/demand_res[R] else demand_res[R]
        
        def is_feas_mach_tsfer(s, avail_comp, avail_extr, demand_vec):
            R = avail_comp.shape[1]
            return (np.sum(avail_comp[s, :] >= demand_vec[0: R]) == R) and (avail_extr >= demand_vec[-1])
        
        def find_best_fit_server_tsfer(avail_comp, avail_extr, demand_vec):
            (S, R), picked_server, norm = avail_comp.shape, 0, np.Infinity
            for s in list(np.random.permutation(range(0, S))):
                if is_feas_mach_tsfer(s, avail_comp, avail_extr, demand_vec):
                    tempd = np.hstack((avail_comp[s, :] - demand_vec[0:R], avail_extr - demand_vec[-1]))
                    if np.linalg.norm(tempd)/np.mean(tempd) < norm:
                        norm = np.linalg.norm(tempd)/np.mean(tempd)
                        picked_server = s
            return picked_server
        
        super().__init__(find_one_task_tsfer_share, is_feas_mach_tsfer, find_best_fit_server_tsfer)


class DRFER_waiting_log(Waiting_log):
    def __init__(self):
        def find_one_task_drfer_share(demanded_res, comp_cap, extr_cap):
            c_sum = np.hstack((np.sum(comp_cap, axis=0), np.array([extr_cap])))
            return np.max(demanded_res/c_sum)
        
        def is_feas_mach_drfer(s, avail_comp, avail_extr, demand_vec):
            R = avail_comp.shape[1]
            return (np.sum(avail_comp[s, :] >= demand_vec[0: R]) == R) and (avail_extr >= demand_vec[-1])
        
        def find_best_fit_server_drfer(avail_comp, avail_extr, demand_vec):
            (S, R), picked_server, norm = avail_comp.shape, 0, np.Infinity
            for s in list(np.random.permutation(range(0, S))):
                if is_feas_mach_drfer(s, avail_comp, avail_extr, demand_vec):
                    tempd = np.hstack((avail_comp[s, :] - demand_vec[0:R], avail_extr - demand_vec[-1]))
                    if np.linalg.norm(tempd)/np.mean(tempd) < norm:
                        norm = np.linalg.norm(tempd)/np.mean(tempd)
                        picked_server = s
            return picked_server
        
        super().__init__(find_one_task_drfer_share, is_feas_mach_drfer, find_best_fit_server_drfer)


class Naive_TSF_waiting_log(Waiting_log):
    def __init__(self):
        def find_one_task_naive_tsf_share(demanded_res, comp_cap, extr_cap):
            virtual_serv = np.hstack((comp_cap, extr_cap))
            (S, R), temp = virtual_serv.shape, 0
            for s in range(0, S):
                temp += np.min(virtual_serv[s, 0:R]/demanded_res[0:R])
            return 1/temp
        
        def is_feas_mach_naive_tsf(s, avail_comp, avail_extr, demand_vec):
            R = avail_comp.shape[1]
            return (np.sum(avail_comp[s, :] >= demand_vec[0: R]) == R) and (avail_extr[s] >= demand_vec[-1])
        
        def find_best_fit_server_naive_tsfer(avail_comp, avail_extr, demand_vec):
            (S, R), picked_server, norm = avail_comp.shape, 0, np.Infinity
            for s in list(np.random.permutation(range(0, S))):
                if is_feas_mach_naive_tsf(s, avail_comp, avail_extr, demand_vec):
                    tempd = np.hstack((avail_comp[s, :] - demand_vec[0: R], avail_extr[s, 0] - demand_vec[-1]))
                    if np.linalg.norm(tempd)/np.mean(tempd) < norm:
                        norm = np.linalg.norm(tempd)/np.mean(tempd)
                        picked_server = s
            return picked_server
        
        super().__init__(find_one_task_naive_tsf_share, is_feas_mach_naive_tsf, find_best_fit_server_naive_tsfer)


class Naive_DRF_waiting_log(Waiting_log):
    def __init__(self):
        def find_one_task_naive_drf_share(demanded_res, comp_cap, extr_cap):
            c_sum = np.hstack((np.sum(comp_cap, axis=0), np.array([np.sum(extr_cap)])))
            return np.max(demanded_res/c_sum)
        
        def is_feas_mach_naive_drf(s, avail_comp, avail_extr, demand_vec):
            R = avail_comp.shape[1]
            return (np.sum(avail_comp[s, :] >= demand_vec[0: R]) == R and
                    avail_extr[s] >= demand_vec[-1])
        
        def find_best_fit_server_naive_drfer(avail_comp, avail_extr, demand_vec):
            (S, R), picked_server, norm = avail_comp.shape, 0, np.Infinity
            for s in list(np.random.permutation(range(0, S))):
                if is_feas_mach_naive_drf(s, avail_comp, avail_extr, demand_vec):
                    tempd = np.hstack((avail_comp[s, :] - demand_vec[0: R], avail_extr[s, 0] - demand_vec[-1]))
                    if np.linalg.norm(tempd)/np.mean(tempd) < norm:
                        norm = np.linalg.norm(tempd)/np.mean(tempd)
                        picked_server = s
            return picked_server
        
        super().__init__(find_one_task_naive_drf_share, is_feas_mach_naive_drf, find_best_fit_server_naive_drfer)
