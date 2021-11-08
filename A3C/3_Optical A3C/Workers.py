
import torch.optim as optim
import ImgProc as proc
from Model_A3C import Net
from A3C_utils import *
import csv
import psutil
from multiprocessing import Value
from ctypes import c_bool
import threading
import cv2 as cv

MAX_EP = 100000
MAX_EP_TIME = 30
UPDATE_GLOBAL_ITER=10

#seed=1

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def Worker(lock,counter, id,shared_model,args,csvfile_name,server,PID):
        name='w%i' % id
        ip='127.0.0.' + str(id + 1)
        lnet = Net(1,7).double()           # local network
        torch.manual_seed(args.seed + id)
        optimizer = optim.Adam(shared_model.parameters(), lr=0.0001)
        lnet.train()
        done=True
        num_ep=0
        point = np.empty([3], dtype=np.float32)
        point[0], point[1], point[2] = 20, 20, -20
        client=first_start(ip)
        prev = np.zeros((128, 128))
        imgOF = np.zeros((128, 128))

        while num_ep < MAX_EP:
            print (name+'--> Episiodio nº: '+str(num_ep))

            client_start(client)
            ###########   INICIO EPISODIO  ############################
            lnet.load_state_dict(shared_model.state_dict())

            if done:
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
            else:
                cx = cx.detach()
                hx = hx.detach()

            values = []
            log_probs = []
            rewards = []
            entropies = []

            total_step = 1
            client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 4, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
            time.sleep(1)

            #####################   STEPS  ############################

            start_time=time.time()
            t=0
            while t <= MAX_EP_TIME and total_step <= 200:
                # Observe new state
                img, state,w,h = proc.get_image(client)
                imgOF=Opticalflow(prev, img)
                cv.imshow('img',img)
                prev=img

                data = client.getMultirotorState()
                position = [data.kinematics_estimated.position.x_val, data.kinematics_estimated.position.y_val,
                            data.kinematics_estimated.position.z_val]

                delta = np.array(point - position, dtype='float32')
                value,policy,(hx,cx) = lnet((state,torch.tensor([delta]),(hx ,cx)))

                prob = F.softmax(policy, dim=-1) #Eliminar negativos con exponenciales y la suma de todo sea 1.
                log_prob = F.log_softmax(policy, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)  # .sum-> Suma los valores de todos los elementos
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()  #detach-> no further tracking of operations. No more gradients
                log_prob = log_prob.gather(1, action)

                collision_info = client.simGetCollisionInfo()
                quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity

                quad_offset=interpret_action(action)

                client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1],quad_vel.z_val+quad_offset[2], 2)


                reward, Remaining_Length = Compute_reward(img, collision_info, point, position, 1)

                done = isDone(reward, collision_info, Remaining_Length)
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                with lock:
                    counter.value += 1
                    csvopen = open(csvfile_name, 'a', newline='')
                    csvfile = csv.writer(csvopen, delimiter=';')
                    inf=str(total_step)+'-> '+str(t)
                    csvfile.writerow([time.time(),name,num_ep,inf,value.item(),log_prob.item(),round(reward,2),round(Remaining_Length,2),point,np.around(position,decimals=2),action.item(),str(collision_info.has_collided)])

                total_step += 1
                ep_time = time.time()
                t=(ep_time - start_time)
                if done:
                    break


            with lock:
                if num_ep % 10 == 0:
                    torch.save(lnet.state_dict(),'Weights_' + str(num_ep) + '.pt')



            if not done:
                value, _, _ = lnet((state,torch.tensor([delta]), (hx, cx)))
                R = value.detach()
                
            R = torch.zeros(1, 1)
            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            maxim = len(rewards)
            if maxim<=20:
                inf=0
            else:
                inf=len(rewards)-20
            for i in reversed(range(inf,maxim-1)):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
                gae = gae * args.gamma * args.gae_lambda + delta_t
                policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]
                #print (R.item(),advantage.item(),value_loss.item(),gae.item(),delta_t.item(),policy_loss.item())
            optimizer.zero_grad()
            #print (policy_loss + args.value_loss_coef * value_loss)
            (policy_loss + args.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(lnet.parameters(), 20 )

            ensure_shared_grads(lnet, shared_model)
            optimizer.step()
            num_ep+=1
            #a.join()
