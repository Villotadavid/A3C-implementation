import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from A3C.pytorch_A3C_F.utils import v_wrap, set_init, push_and_pull, record
from A3C.pytorch_A3C_F.shared_adam import SharedAdam
import gym
import math, os
from Workers import Worker
from Model_A3C import Net
import Model


os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000



if __name__ == "__main__":
    gnet = Model.DQN()          # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
                                # Comparte la dirección de memoria para todos los procesos

    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count()-3)]
    print (workers)
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()
