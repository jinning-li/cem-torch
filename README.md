# Cross Entropy Method - Pytorch

The Cross Entropy Method (CEM) is implemented under Pytorch in this repo. 
It is especially useful for receding horizon style model predictive control. 
This CEM solver can also be plugged into some model-based reinforcement learning algorithms when a deep neural net policy is not required and trained. 

The CEM is a Monte Carlo method for optimization. 
This repo is for continuous problems. 
Specifically, this CEM solver can solve the optimization problem as below:
$
\begin{aligned}
    \min_a \; & \mathcal{C}_{f}(s_H) + \sum_{k=0}^{H-1} \mathcal{C}(s_k, a_k) \\
    \text{s.t.}\;\;&s_{k+1} = f(s_k, a_k), \; k=\{0, 1, \dots, H-1\} \\
    &s_0 = s_{\text{init}}
\end{aligned}
$
where $s_t$ and $a_t$ are the states and actions at time step $k$, $s_\text{init}$ is the initial condition. $H$ is the horizon, $\mathcal{C}(\cdot)$ is the cost function, $\mathcal{C}_f(\cdot)$ is the terminal cost function, and $f(\cdot)$ is the dynamics function. 

Please get started from the example usage in `test.py`. More detailed examples are to be uploaded. 

If you find this repo useful, please consider citing our paper:
```
@ARTICLE{li2022hierarchical,
  author={Li, Jinning and Tang, Chen and Tomizuka, Masayoshi and Zhan, Wei},
  journal={IEEE Robotics and Automation Letters}, 
  title={Hierarchical Planning Through Goal-Conditioned Offline Reinforcement Learning}, 
  year={2022},
  volume={7},
  number={4},
  pages={10216-10223},
  doi={10.1109/LRA.2022.3190100}}
```