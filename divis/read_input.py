import sys
import numpy as np
from os import path, makedirs
from time import sleep
from random import randint


def read_argv(ch, ro):
    # arg 1: number of server triple
    # arg 2: mem to cpu ratio [0, 1]
    # arg 3: server_seed 0..10000
    sr = 1000
    samp_rate = sr / 1000.
    end_part = 1
    rho = ro / 10.0
    num_server = int(sys.argv[1])
    channel_bw = ch * 1.0e6
    mem_to_cpu_ratio = float(sys.argv[2])
    serv_seed = int(sys.argv[3])
    c = np.array([[1.0, 1.0 * mem_to_cpu_ratio] for _ in range(num_server)] +
                 [[1.0 - rho, (1.0 + rho) * mem_to_cpu_ratio] for _ in range(num_server)] +
                 [[1.0 + rho, (1.0 - rho) * mem_to_cpu_ratio] for _ in range(num_server)])
    server_dict = {'c': c, 'num_server': 3*num_server, 'server_seed': serv_seed,
                   'mem_to_cpu_ratio': mem_to_cpu_ratio, 'bw_MHz': ch, 'rho': rho}
    location_dict = {'trace_path': "/scratch/meskarer/clusterdata-2011-2",
                     'scratch_path': "/scratch/meskarer"}
    event_reading_dict = {'jump': 1 , 'start_part': 1 , 'end_part': end_part,
                          'sample_rate': samp_rate, 'server_seed': serv_seed}
    scr_path = path.abspath(path.join(path.dirname(__file__), '..'))
    RW_path = path.join(scr_path, 'google_divis', f"{sr}perkilo_{num_server * 3}serv_{ch}MHz_{ro}rho_{end_part}part_{int((mem_to_cpu_ratio*100)//1)}_perc_mem_to_cpu_ratio_{serv_seed}server_seed")
    location_dict['RW_path'] = RW_path
    return location_dict, event_reading_dict, server_dict, channel_bw
