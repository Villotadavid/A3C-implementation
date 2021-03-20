import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import v_wrap, set_init, push_and_pull, record
from A3C_utils import *
import gym
import math, os
from Workers import Worker
from Model_A3C import Net
import time
import csv
import argparse


os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')



if __name__ == "__main__":

    num_workers=3
    seed=1
    torch.manual_seed(seed)

    args = parser.parse_args()

    shared_model = Net(1,5).double()
    shared_model.share_memory()

    counter = mp.Value('i', 0)
    lock = mp.Lock()
    loop_finish=mp.Manager().list([False]*num_workers)
    print ( not all( element for element in loop_finish))
    processes = []
    name=0
    File=True
    log=[]
    while File:
        File=os.path.exists('Training_data_'+str(name)+'.csv')

        if File:
            pass
        else:
            csv_file='Training_data_'+str(name) + '.csv'
            csvopen = open('Training_data_' + str(name) + '.csv', 'a', newline='')
            csvfile = csv.writer(csvopen, delimiter=';')
            csvfile.writerow(['Time','Hilo', 'Episodio', 'Step', 'Values', 'log_prob', 'Rewards', 'Remaining_Length', 'Point', 'Position','Action','%CPU','%Memoria'])
        name += 1

    for i in range (0,num_workers):
        create_env(i)

    for name in range(0, num_workers):
        p = mp.Process(target=Worker, args=(lock,counter, name,shared_model,args,csv_file,loop_finish))
        time.sleep(2)
        p.start()
        processes.append(p)
    [p.join() for w in processes]



