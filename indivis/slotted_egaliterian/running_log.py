import bisect as bs


class Running_log:
    def __init__(self):
        self.running_job_dct = {}  # {str(job_id): Job}
        self.j_t_idx_sorted_by_tent_fin_time = []

    def update_allocated_task_and_tent_fin(self, waiting_log, time_current):
        job = self.running_job_dct[waiting_log.picked_job_id]
        task = job.waiting_task_dct.pop(waiting_log.picked_task_idx)
        task.last_time_stamp_added_to_running_task_dct = time_current
        task.waited_time += time_current - task.last_time_stamp_added_to_waiting_task_dct
        task.allocated_server = waiting_log.picked_server
        job.active_fair_share += job.one_task_fair_share
        job.running_task_dct[waiting_log.picked_task_idx] = task
        task_tent_fin_time = time_current + job.one_task_duration - task.executed_time
        # place [task_tent_fin_time, [[waiting_log.picked_job_id, waiting_log.picked_task_idx]]]
        # appropariately in self.j_t_idx_sorted_by_tent_fin_time
        i = bs.bisect_left([x[0] for x in self.j_t_idx_sorted_by_tent_fin_time], task_tent_fin_time)
        if i == len(self.j_t_idx_sorted_by_tent_fin_time):
            self.j_t_idx_sorted_by_tent_fin_time.append([task_tent_fin_time, [[waiting_log.picked_job_id, waiting_log.picked_task_idx]]])
        elif self.j_t_idx_sorted_by_tent_fin_time[i][0] == task_tent_fin_time:
            self.j_t_idx_sorted_by_tent_fin_time[i][1].append([waiting_log.picked_job_id, waiting_log.picked_task_idx])
        else:
            self.j_t_idx_sorted_by_tent_fin_time[i:i] = [[task_tent_fin_time, [[waiting_log.picked_job_id, waiting_log.picked_task_idx]]]]