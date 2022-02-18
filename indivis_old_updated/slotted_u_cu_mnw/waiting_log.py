import math
import numpy as np
from random import choice
from .job import Task, Job
import cvxpy as cpy

class Waiting_log:
    def __init__(self, gamma_finder, relaxed_opt_prob):
        self.waiting_job_dct = {}  # {str(job_id): Abs_sub_job}
        self.can_allocate = False
        self.picked_job_id = ''
        self.picked_server = 0
        self.picked_task_idx = 0
        self.picked_task_arrival_time = 0.0
        self.picked_task_executed_time = 0.0
        self.picked_job_over = False
        self.it_is_a_resubmitted_task = False
        # define the function that checks if a machine is feasible for a task
        def feasiblity_check_func(s, avail_comp, avail_extr, demand_vec):
            R = avail_comp.shape[1]
            return (np.sum(avail_comp[s, :] >= demand_vec[0: R]) == R and 
                    avail_extr >= demand_vec[-1])
        self.is_feasible_on_mach = feasiblity_check_func
        self.gamma_finder = gamma_finder
        self.relaxed_opt_prob = relaxed_opt_prob

    def create_and_add_jobs_to_wait_job_dct(self, arrived_job, serv_cap, extr_cap):
        num_arrived_task = arrived_job.shape[0]
        temp_pointer = 0
        while temp_pointer < num_arrived_task:
            arr_job_id = arrived_job.loc[temp_pointer, 'job_id']
            if arr_job_id not in self.waiting_job_dct:
                demand_vec = arrived_job.loc[temp_pointer, ['req_cpu_core',
                                                            'req_mem_GB',
                                                            'req_bw']].to_numpy()
                self.waiting_job_dct[arr_job_id] =\
                    Job(arrived_job.loc[temp_pointer, 'int_runtime_sec'],
                        arrived_job.loc[temp_pointer, 'int_sub_time'],
                        demand_vec,
                        self.gamma_finder(serv_cap, extr_cap, demand_vec),
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
                        append([self.waiting_job_dct[arr_job_id].\
                            waiting_task_dct[int_id].executed_time,
                                int_id])
                temp_pointer += 1

    def allocate_from_waiting_log(self, total_comp_cap, total_extr_cap):
        N = len(self.waiting_job_dct)
        self.can_allocate = False if N == 0 else True
        if N == 0: return [], []
        R = total_comp_cap.shape[1] + np.size(total_extr_cap)
        job_id_list = []
        demand_matrix = np.zeros(shape=(N, R))
        max_num_task = np.zeros(shape=(N, 1))
        gamma = np.zeros(shape=(N, 1))
        row = 0
        for job_id, job in self.waiting_job_dct.items():
            job_id_list.append(job_id)
            demand_matrix[row, :] = job.demanded_res
            max_num_task[row, 0] = len(job.waiting_task_dct)
            gamma[row, 0] = job.gamma
            row += 1
        if N == 1:
            tmp = np.min(total_comp_cap/demand_matrix[0:1, :-1], axis=1, keepdims=True).astype('int')
            x_max = min(np.sum(tmp),
                        int(total_extr_cap/demand_matrix[0, -1]),
                        max_num_task[0, 0])
            x_round = np.zeros(shape=(1, total_comp_cap.shape[1]))
            xsum = 0
            for s in range(x_round.shape[1]):
                if xsum + tmp[s, 0] > x_max:
                    x_round[0, s] = x_max - xsum
                    break
                else:
                    x_round[0, s] = tmp[s, 0]
                    xsum += tmp[s, 0]
            return job_id_list, x_round.astype('int')
        else:
            x = self.relaxed_opt_prob(total_comp_cap, total_extr_cap,
                                      demand_matrix, max_num_task, gamma)
            self.it_is_a_resubmitted_task = False
            # Find rounded down number of tasks
            x_round = np.array(np.floor(x), dtype=np.int)
            return job_id_list, x_round

    def get_Q_length(self):
        queueLength = 0
        for job in self.waiting_job_dct.values():
            queueLength += len(job.waiting_task_dct)
        return queueLength


class MNW_waiting_log(Waiting_log):
    def __init__(self):
        def gamma_finder(comp_cap, extr_cap, demand):
            gamma_comp = np.sum(np.min(comp_cap / demand[0:comp_cap.shape[1]], axis=1))
            gamma_comm = np.min(extr_cap / demand[comp_cap.shape[1]:])
            return min(gamma_comp, gamma_comm)

        def mnw_relaxed_opt(comp_cap, extr_cap, demand, max_num_tasks, gamma):
            N = demand.shape[0]
            (S, R_comp) = comp_cap.shape
            R_extr = np.size(extr_cap)
            x = cpy.Variable(shape=(N, S), nonneg=True)
            const = []
            # Capacity constraint
            const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
            const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
            const += [np.ones(shape=(1, S)) @ x.T <= max_num_tasks.T]
            for i in range(N):
                v = (comp_cap < demand[i:i+1, 0:R_comp]).any(axis=1).astype('float')
                if sum(v) > 0:
                    const += [x[i:i+1, :] @ v == 0]
            obj = cpy.Maximize(cpy.sum(cpy.log(x @ np.ones(shape=(S, 1)))))
            prob = cpy.Problem(obj, const)
            prob.solve(solver=cpy.SCS, verbose=False)
            return x.value
        super().__init__(gamma_finder, mnw_relaxed_opt)


class Utilitarian_waiting_log(Waiting_log):
    def __init__(self):
        def gamma_finder(comp_cap, extr_cap, demand):
            gamma_comp = np.sum(np.min(comp_cap / demand[0:comp_cap.shape[1]], axis=1))
            gamma_comm = np.min(extr_cap / demand[comp_cap.shape[1]:])
            return min(gamma_comp, gamma_comm)

        def u_relaxed_opt(comp_cap, extr_cap, demand, max_num_tasks, gamma):
            N = demand.shape[0]
            (S, R_comp) = comp_cap.shape
            R_extr = np.size(extr_cap)
            x = cpy.Variable(shape=(N, S), nonneg=True)
            const = []
            # Capacity constraint
            const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
            const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
            const += [np.ones(shape=(1, S)) @ x.T <= max_num_tasks.T]
            obj = cpy.Maximize(np.ones(shape=(1, N)) @ x @ np.ones(shape=(S, 1)))
            prob = cpy.Problem(obj, const)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            return x.value
        super().__init__(gamma_finder, u_relaxed_opt)


class Relative_utilitarian_waiting_log(Waiting_log):
    def __init__(self):
        def gamma_finder(comp_cap, extr_cap, demand):
            gamma_comp = np.sum(np.min(comp_cap / demand[0:comp_cap.shape[1]], axis=1))
            gamma_comm = np.min(extr_cap / demand[comp_cap.shape[1]:])
            return min(gamma_comp, gamma_comm)

        def ru_relaxed_opt(comp_cap, extr_cap, demand, max_num_tasks, gamma):
            N = demand.shape[0]
            (S, R_comp) = comp_cap.shape
            R_extr = np.size(extr_cap)
            x = cpy.Variable(shape=(N, S), nonneg=True)
            const = []
            # Capacity constraint
            const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
            const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
            const += [np.ones(shape=(1, S)) @ x.T <= max_num_tasks.T]
            obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
            prob = cpy.Problem(obj, const)
            prob.solve(solver=cpy.MOSEK, verbose=False)
            return x.value
        super().__init__(gamma_finder, ru_relaxed_opt)


class Cautious_utilitarian_waiting_log(Waiting_log):
    def __init__(self):
        def gamma_finder(comp_cap, extr_cap, demand):
            gamma_comp = np.sum(np.min(comp_cap / demand[0:comp_cap.shape[1]], axis=1))
            gamma_comm = np.min(extr_cap / demand[comp_cap.shape[1]:])
            return min(gamma_comp, gamma_comm)

        def cu_relaxed_opt(comp_cap, extr_cap, demand, max_num_tasks, gamma):
            N, (S, R_comp)  = demand.shape[0], comp_cap.shape
            R_extr = np.size(extr_cap)
            delta = np.zeros(shape=(N, N))
            for u1 in range(N):
                for u2 in range(N):
                    delta[u1, u2] = np.min(demand[u1, :] / demand[u2, :])
            # Find users that must be saturated
            not_sat_user = []
            sat_user = []
            for u in range(N):
                if max_num_tasks[u, 0] > (gamma[u, 0] / N):
                    not_sat_user.append(u)
                else:
                    sat_user.append(u)
            x_value = np.zeros(shape=(N, S))
            if len(sat_user) == N:
                x = cpy.Variable(shape=(N, S), nonneg=True)
                const = []
                # Capacity constraint
                const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                for u in range(N):
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
                prob = cpy.Problem(obj, const)
                prob.solve(solver=cpy.MOSEK, verbose=False)
                return x.value
            rep_cntr = 0
            while not (extr_cap - np.ones(shape=(1, S)) @ x_value.T @ demand[:, R_comp:(R_comp + R_extr)] <= 1e-4
                    or (np.min(comp_cap - x_value.T @ demand[:, 0:R_comp], axis=1) <= 1e-4).all()
                    or len(sat_user) == N
                    or rep_cntr > 9):
                rep_cntr += 1
                x = cpy.Variable(shape=(N, S), nonneg=True)
                const = []
                # Capacity constraint
                const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                for u in not_sat_user:
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T <= max_num_tasks[u, 0]]
                for u in sat_user:
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                # SI constraint
                for u in not_sat_user:
                    const += [(gamma[u, 0] / N) <= x[u:u+1, 0:S] @ np.ones(shape=(S, 1))]
                # EF constraint
                for u2 in not_sat_user:
                    for u1 in range(N):
                        const += [(delta[u1, u2] * np.ones(shape=(1, S)) @ x[u1, :] - np.ones(shape=(1, S)) @ x[u2, :].T) <= 0]
                obj = cpy.Maximize(np.ones(shape=(1, N)) @ x @ np.ones(shape=(S, 1)))
                prob = cpy.Problem(obj, const)
                prob.solve(solver=cpy.MOSEK, verbose=False)
                x_value = x.value
                sum_x = np.sum(x_value, axis=1, keepdims=True)
                for u in range(N):
                    if (abs(sum_x[u, 0] - max_num_tasks[u, 0]) < 1e-4 and
                            u in not_sat_user):
                        sat_user.append(u)
                        not_sat_user.remove(u)
            return x.value
        super().__init__(gamma_finder, cu_relaxed_opt)


class Cautious_relative_utilitarian_waiting_log(Waiting_log):
    def __init__(self):
        def gamma_finder(comp_cap, extr_cap, demand):
            gamma_comp = np.sum(np.min(comp_cap / demand[0:comp_cap.shape[1]], axis=1))
            gamma_comm = np.min(extr_cap / demand[comp_cap.shape[1]:])
            return min(gamma_comp, gamma_comm)

        def cru_relaxed_opt(comp_cap, extr_cap, demand, max_num_tasks, gamma):
            N, (S, R_comp)  = demand.shape[0], comp_cap.shape
            R_extr = np.size(extr_cap)
            delta = np.zeros(shape=(N, N))
            for u1 in range(N):
                for u2 in range(N):
                    delta[u1, u2] = np.min(demand[u1, :] / demand[u2, :])
            # Find users that must be saturated
            not_sat_user = []
            sat_user = []
            for u in range(N):
                if max_num_tasks[u, 0] > (gamma[u, 0] / N):
                    not_sat_user.append(u)
                else:
                    sat_user.append(u)
            x_value = np.zeros(shape=(N, S))
            if len(sat_user) == N:
                x = cpy.Variable(shape=(N, S), nonneg=True)
                const = []
                # Capacity constraint
                const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                for u in range(N):
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                for i in range(N):
                    v = (comp_cap < demand[i:i+1, 0:R_comp]).any(axis=1).astype('float')
                    if sum(v) > 0:
                        const += [x[i:i+1, :] @ v == 0]
                obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
                prob = cpy.Problem(obj, const)
                prob.solve(solver=cpy.MOSEK, verbose=False)
                if prob.status == cpy.INFEASIBLE:
                    x = cpy.Variable(shape=(N, S), nonneg=True)
                    const = []
                    # Capacity constraint
                    const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                    const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                    for u in range(N):
                        const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                    obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
                    prob = cpy.Problem(obj, const)
                    prob.solve(solver=cpy.MOSEK, verbose=False)
                return x.value
            rep_cntr = 0
            while not (extr_cap - np.ones(shape=(1, S)) @ x_value.T @ demand[:, R_comp:(R_comp + R_extr)] <= 1e-4
                    or (np.min(comp_cap - x_value.T @ demand[:, 0:R_comp], axis=1) <= 1e-4).all()
                    or len(sat_user) == N
                    or rep_cntr > 9):
                rep_cntr += 1
                x = cpy.Variable(shape=(N, S), nonneg=True)
                const = []
                # Capacity constraint
                const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                for u in not_sat_user:
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T <= max_num_tasks[u, 0]]
                for u in sat_user:
                    const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                # SI constraint
                for u in not_sat_user:
                    const += [(gamma[u, 0] / N) <= x[u:u+1, 0:S] @ np.ones(shape=(S, 1))]
                # EF constraint
                for u2 in not_sat_user:
                    for u1 in range(N):
                        const += [(delta[u1, u2] * np.ones(shape=(1, S)) @ x[u1, :] - np.ones(shape=(1, S)) @ x[u2, :].T) <= 0]
                for i in range(N):
                    v = (comp_cap < demand[i:i+1, 0:R_comp]).any(axis=1).astype('float')
                    if sum(v) > 0:
                        const += [x[i:i+1, :] @ v == 0]
                obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
                prob = cpy.Problem(obj, const)
                prob.solve(solver=cpy.MOSEK, verbose=False)
                if prob.status == cpy.INFEASIBLE:
                    x = cpy.Variable(shape=(N, S), nonneg=True)
                    const = []
                    # Capacity constraint
                    const += [x.T @ demand[:, 0:R_comp] <= comp_cap]
                    const += [np.ones(shape=(1, S)) @ x.T @ demand[:, R_comp:(R_comp + R_extr)] <= extr_cap]
                    for u in not_sat_user:
                        const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T <= max_num_tasks[u, 0]]
                    for u in sat_user:
                        const += [np.ones(shape=(1, S)) @ x[u:u+1, 0:S].T == max_num_tasks[u, 0]]
                    # SI constraint
                    for u in not_sat_user:
                        const += [(gamma[u, 0] / N) <= x[u:u+1, 0:S] @ np.ones(shape=(S, 1))]
                    # EF constraint
                    for u2 in not_sat_user:
                        for u1 in range(N):
                            const += [(delta[u1, u2] * np.ones(shape=(1, S)) @ x[u1, :] - np.ones(shape=(1, S)) @ x[u2, :].T) <= 0]
                    obj = cpy.Maximize((1 / gamma).T @ x @ np.ones(shape=(S, 1)))
                    prob = cpy.Problem(obj, const)
                    prob.solve(solver=cpy.MOSEK, verbose=False)
                x_value = x.value
                sum_x = np.sum(x_value, axis=1, keepdims=True)
                for u in range(N):
                    if (abs(sum_x[u, 0] - max_num_tasks[u, 0]) < 1e-4 and
                            u in not_sat_user):
                        sat_user.append(u)
                        not_sat_user.remove(u)
            return x.value
        super().__init__(gamma_finder, cru_relaxed_opt)
