# A3C-implementation
In this work, we apply artificial intelligence to guide a drone to a certain point autonomously. Unreal engine creates a virtual environment where the drone can fly, and the algorithm is trained simulating the drone dynamics thanks to Airsim plugin. The implemented algorithm is Asynchronous Actor-Critic Advantage (A3C), which trains a neural network with less computing resources than standard reinforcement learning algorithms that normally needs costly GPUs. To prove these advantages, several experiments are run using a different number of parallel simulations (threads). The drone should reach a point randomly generated each episode. The reward, the value and the advantage function are used to evaluate the performance. 

As expected, these experiments show that a higher number of threads helps the leaning process improve and become more stable. These learning results are of interest to optimize the computing resources in future applications.

## Multithreading in Airsim

For A3C implementation, several Airsim instances will be used at the same time; each one of them will be one of the environments in which the agent will interact with. 

As there will be several threads, several instances of Airsim will have to be implemented at the same time. This is one of the main difficulties, as the configuration of Airsim does not allow multithreading. The only way we have figured out to solve that is to configure different Local Host IPs to each Airsim instance. This way, each thread will have its environment with its single IP. The configured IPs are from 127.0.0.1 to 127.0.0.n, being n the number of threads used in the simulation

<img src="A3C/images/path3726-37.png">

## Random point generation

The objective of the drone is to reach a specific point in a 3D virtual environment. The random point will be calculated using trigonometry in XY plane (α), and a random Z coordinate (z) in a constant spherical radius d, as the following image illustrates.

<img src="A3C/images/g9299.png">

## Reward function

The reward function will determine how good is the action yielded by the neural network once it is carried out. In other words, it is the way we can tell the algorithm how good it is performing. 
The objective of the drone is to reach a given point in the environment. Thus, the variable used to compute the reward function will be the distance between the absolute position of the drone and the absolute coordinates of the objective point 

<img src="A3C/images/Reward1.png">

## Training results
### Final episodes

If we pay attention to the behaviour in the final episodes (2400), we can see (Figure 7) that it behaves properly.
In the initial steps the Value function increases as the remaining length decreases. This means that at the time we are getting nearer to the objective point, the state we are in is better than before, and given the policy output by the network and the state, the expected Reward is larger. When the Remaining length starts growing in the middle steps, the Value function stops increasing and starts decreasing as the drone is getting further to the objective point.

<img src="A3C/images/EP2400.png">

Following an image with the 3D representation of a random episode

<img src="A3C/images/Trake.png">

## A3C thread number comparation

To evaluate how a simulation has learned, we will use the Value function as it is the one that tells us how good the states along the simulation have been. Since each single step of each episode has its own value, we will calculate the mean function of each episode and afterwards a line will be fitted in the points to ease the understanding. The graphs in Figure  10 contains the values for the different simulations with different number of threads.
The simulation with one single thread presents a stable behaviour during the training and it also reaches the lowest values. However, the simulation with 6 threads is the fastest one to reach the highest values. It only needs some hundreds of episodes to reach the same value than the while simulation with 2 threads. 
For the simulations with 2, 3 and 5 threads, we can see that each one surpasses the values of the previous one.
This is the expected behaviour as mentioned in. Thus, for our implementation, we can also conclude that to have many threads directly affects the learning stability. The simulation with one single thread behaves significantly worse than those with more than one thread. The value oscillates in the first two thousand steps. Also, the more threads used the bigger values attained in the simulation and it also learns faster in the initial epochs.

<img src="A3C/images/rplotmax.png">

