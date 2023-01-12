import time
import torch
from torch import nn
from cem import CEM

def dyn(obs, ac):
    """An example dynamics function.
    
    This can also be a torch.nn.Module like the cost function below.
    """
    return 2 * obs + torch.sum(ac)

class CostFn(nn.Module):
    """An example cost function."""
    def forward(self, obs, ac):
        res = torch.sum((obs-1)**2, dim=-1) * torch.sum((ac+1)**2, dim=-1)
        return res.flatten()

def main():
    obs_dim = 46
    ac_dim = 26
    batch_size = 32
    cost_fn = CostFn()

    solver = CEM(
        obs_dim,
        ac_dim, 
        dyn,
        cost_fn,
        ac_lb=torch.tensor(-1.),
        ac_ub=torch.tensor(1.),
        num_samples=100, 
        num_iterations=3, 
        num_elite=10, 
        horizon=5,
        init_cov_diag=1.,
        device="cpu",
    )

    obs = torch.rand((batch_size, obs_dim))

    start_t = time.time()
    ac, log_probs = solver.solve(obs, get_log_probs=True)
    end_t = time.time()

    print(f"The shape of action solution: {ac.shape}")
    print(f"The shape of log probs: {log_probs.shape}")
    print(f"Process time: {end_t - start_t}s")

if __name__=="__main__": 
    main()
