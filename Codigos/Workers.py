
import torch.optim as optim
import time
import ImgProc as proc
import ReinforceLearning as RL
from Model_A3C import Net
from A3C_utils import *
import csv

MAX_EP = 5
MAX_EP_STEP = 100
UPDATE_GLOBAL_ITER=10
seed=1

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def Worker(lock,counter, id,shared_model,args,csvfile_name):
        name='w%i' % id
        lnet = Net(1,5).double()           # local network
        torch.manual_seed(args.seed + id)

        client, Process = create_env(id)
        total_step = 1
        done=True
        optimizer = optim.Adam(shared_model.parameters(), lr=0.0001)
        point = np.empty([3], dtype=np.float32)
        point[0], point[1], point[2] = 30, 15, -20
        num_ep=0
        while num_ep < MAX_EP:
            print (name+'--> Episiodio nº: '+str(num_ep))
            client_start(client)
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

            client.moveToPositionAsync(int(point[0]), int(point[1]), int(point[2]), 4, 3e+38,airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0))
            time.sleep(2)
            #####################   STEPS  ############################
            log_data=[]
            for t in range(MAX_EP_STEP):
                # Observe new state
                img, state = proc.get_image(client)
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

                quad_offset,angular=interpret_action(action)

                if not angular:
                    client.moveByVelocityAsync( quad_offset[0],  quad_offset[1],quad_offset[2], 2)
                else:
                    client.rotateToYawAsync(quad_offset[2],3e+38,5)
                    #client.moveByVelocityAsync(quad_vel.x_val,quad_vel.z_val,quad_vel.z_val, 2)

                reward, Remaining_Length = RL.Compute_reward(img, collision_info, point, position, 1)

                done = isDone(reward, collision_info, Remaining_Length)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)

                with lock:
                    counter.value += 1

                if done:

                    break

                log_data.append([time.time(),name,num_ep,t,value.item(),log_prob.item(),round(reward,2),round(Remaining_Length,2),point,np.around(position,decimals=2),action.item()])

                total_step += 1


            with lock:
                csvopen = open(csvfile_name, 'w', newline='')
                csvfile = csv.writer(csvopen, delimiter=';')
                csvfile.writerows(log_data)
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

        Process.terminate()
