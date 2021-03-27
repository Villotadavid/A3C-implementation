
import torch.optim as optim
import ImgProc as proc
import ReinforceLearning as RL
from Model_A3C import Net
from A3C_utils import *
import csv
import psutil

MAX_EP = 100000
MAX_EP_STEP = 200
UPDATE_GLOBAL_ITER=10
#seed=1

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def Worker(lock,counter, id,shared_model,args,csvfile_name,loop_finish):
        name='w%i' % id
        ip='127.0.0.' + str(id + 1)
        lnet = Net(1,5).double()           # local network
        #torch.manual_seed(args.seed + id)
        optimizer = optim.Adam(shared_model.parameters(), lr=0.0001)

        point = np.empty([3], dtype=np.float32)
        point[0], point[1], point[2] = 30, 15, -20

        total_step = 1
        done=True
        num_ep=0

        while num_ep < MAX_EP:
            print (name+'--> Episiodio nº: '+str(num_ep))

            client=client_start(ip)
            ###########   INICIO EPISODIO  ############################
            lnet.load_state_dict(shared_model.state_dict())
            #Hay que hacer un reset al environment

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

            #client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 4, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
            #time.sleep(2)
            #####################   STEPS  ############################
            log_data=[]
            loop_finish[id] = False

            for t in range(MAX_EP_STEP):
                # Observe new state
                img, state,w,h = proc.get_image(client)
                data = client.getMultirotorState()
                position = [data.kinematics_estimated.position.x_val, data.kinematics_estimated.position.y_val,
                            data.kinematics_estimated.position.z_val]

                delta = np.array(point - position, dtype='float32')
                value,policy,(hx,cx) = lnet.forward( (state,torch.tensor([delta]),(hx ,cx)))

                collision_info = client.simGetCollisionInfo()

                prob = F.softmax(policy, dim=-1) #Eliminar negativos con exponenciales y la suma de todo sea 1.
                log_prob = F.log_softmax(policy, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)  # .sum-> Suma los valores de todos los elementos
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).detach()  #detach-> no further tracking of operations. No more gradients
                log_prob = log_prob.gather(1, action)

                quad_vel = client.getMultirotorState().kinematics_estimated.linear_velocity

                quad_offset=interpret_action(action)

                client.moveByVelocityAsync(quad_vel.x_val+quad_offset[0], quad_vel.y_val+quad_offset[1],quad_vel.z_val+quad_offset[2], 2)


                reward, Remaining_Length = RL.Compute_reward(img, collision_info, point, position, 1)

                done = isDone(reward, collision_info, Remaining_Length)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    break

                memoria=psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
                log_data.append([time.time(),name,num_ep,t,value.item(),log_prob.item(),round(reward,2),round(Remaining_Length,2),point,np.around(position,decimals=2),action.item(),str(collision_info.has_collided) ,psutil.cpu_percent(),memoria,w,h])

                with lock:
                    counter.value += 1
                    csvopen = open(csvfile_name, 'a', newline='')
                    csvfile = csv.writer(csvopen, delimiter=';')
                    #csvfile.writerows(log_data)
                    csvfile.writerow([time.time(),name,num_ep,t,value.item(),log_prob.item(),round(reward,2),round(Remaining_Length,2),point,np.around(position,decimals=2),action.item(),str(collision_info.has_collided),psutil.cpu_percent(),memoria,w,h])

                total_step += 1

            with lock:
                if num_ep % 10 == 0:
                    torch.save(lnet.state_dict(),'Weights_' + str(num_ep) + '.pt')


            #loop_finish[id]=True
            #check_loop_finish(loop_finish)

            R = torch.zeros(1, 1)

            if not done:
                value, _, _ = lnet((state,torch.tensor([delta]), (hx, cx)))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
                gae = gae * args.gamma * args.gae_lambda + delta_t

                policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

            optimizer.zero_grad()

            (policy_loss + args.value_loss_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(lnet.parameters(), args.max_grad_norm)

            ensure_shared_grads(lnet, shared_model)
            optimizer.step()
            num_ep+=1

