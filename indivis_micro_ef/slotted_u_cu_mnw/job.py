from numpy import DataSource


class Task:
    def __init__(self, arrival_time):
        self.task_arrival_time = arrival_time
        self.last_time_stamp_added_to_waiting_task_dct = arrival_time
        self.executed_time = 0.0
        self.last_time_stamp_added_to_running_task_dct = -1.0
        self.waited_time = 0.0
        self.allocated_server = -1
        self.finish_time = -1.0


class Job:
    def __init__(self, one_task_duration, first_task_arrival_time, demanded_res,
                gamma, mice, sta_t, sta_jct):
        """
        self.task_idx_sorted_by_executed_time only contains tasks in waiting_task_dct.
        self.task_idx_sorted_by_executed_time[0] has the minimum executed time.
        To allocate tasks on server, we prefer to pick the task which is most of
        it execution is over, i.e. self.task_idx_sorted_by_executed_time[-1]
        self.task_idx_sorted_by_executed_time = [[executed_time, task_idx], ...]
        """
        self.waiting_task_dct = {}
        self.running_task_dct = {}
        self.finished_task_dct = {}
        self.one_task_duration = one_task_duration
        self.job_arrival_time = first_task_arrival_time
        self.demanded_res = demanded_res
        self.task_idx_sorted_by_executed_time = []
        self.gamma = gamma
        self.mice = mice
        self.sta_t = sta_t
        self.sta_jct = sta_jct
        self.sum_min_ef_satsf = 0.0
        self.sum_si_satsf = 0.0
