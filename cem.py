"""The Cross Entropy Method implemented with Pytorch.
"""
from typing import Union, Callable, Optional
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

OBS_SHAPE_ERROR =  "obs.shape must be either (obs_dim, ) or (batch_size, obs_dim)!"
AC_BOUND_ERROR = "Both ac_lb and ac_ub must be provided!"

def batch_cov(batch: torch.Tensor) -> torch.Tensor:
    """Compute covariance matrix with batch dimension.

    Args:
        batch: shape(batch_size, N_samples, data_dim)
    Returns:
        cov: shape(batch_size, data_dim, data_dim)
    """
    num_samples = batch.shape[1]
    centered = batch - batch.mean(dim=1, keepdim=True)
    prods = torch.einsum('bni, bnj->bij', centered, centered)
    cov = prods / (num_samples - 1) 
    return cov


class CEM:
    """An optimization solver based on Cross Entropy Method under Pytorch. 

    This CEM solver supports batch dimension of observations, and can solve receding
    horizon style model predictive control problems. If only one step cost is to be
    considered, just set horizon to be one. 
    """
    def __init__(
        self, 
        obs_dim: int, 
        ac_dim: int, 
        dynamics_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        ac_lb: torch.Tensor = None,
        ac_ub: torch.Tensor = None,
        terminal_obs_cost: Callable[[torch.Tensor], torch.Tensor] = None,
        num_samples: int = 100, 
        num_iterations: int = 3, 
        num_elite: int = 10, 
        horizon: int = 15,
        init_cov_diag: float = 1.,
        device: str = "cpu",
    ):
        """Construct all necessary attributes.

        Args:
            obs_dim: the observation (state) dimension.
            ac_dim: the action dimension.
            dynamics_fn: the dynamics function [obs, ac] -> [next_obs]. It is for doing
                rollouts when computing the total cost of a trajectory of H horizon.
            cost_fn: the cost function [obs, ac] -> cost.
            ac_lb: the lower bound of the action space.
            ac_ub: the upper bound of the action space.
            terminal_obs_cost: the function to compute the cost of terminal observation.
            num_samples: the total number of samples generated in each iteration of CEM.
            num_iterations: the number of iterations of the CEM.
            num_elite: the number of top samples to be selected in each iteration of CEM.
            horizon: the horizon of the trajectory to be considered for computing cost.
            init_cov_diag: the diagonal element of the initial covirance of the action.
            device: the device being used by torch.
        """
        self._device = device
        self._dtype = torch.float32  
        
        self._num_samples = num_samples  
        self._horizon = horizon  
        self._num_iter = num_iterations
        self._num_elite = num_elite

        self._obs_dim = obs_dim
        self._ac_dim = ac_dim
        self._batch_size = None
        self._ac_lb = ac_lb
        self._ac_ub = ac_ub
        assert self._check_bound(), AC_BOUND_ERROR

        self._dyn = dynamics_fn
        self._cost_fn = cost_fn
        self._terminal_obs_cost = terminal_obs_cost

        self._ac_dist = None
        self._init_cov_diag = init_cov_diag
        self._cov_reg = torch.eye(self._horizon * self._ac_dim, device=self._device, 
            dtype=self._dtype).unsqueeze(0) * init_cov_diag * 1e-5

    def _init_ac_dist(self):
        assert self._batch_size is not None
        mean = torch.zeros((self._batch_size, self._horizon * self._ac_dim), 
            device=self._device, dtype=self._dtype)
        cov = self._init_cov_diag * torch.eye(self._horizon * self._ac_dim, 
            device=self._device, dtype=self._dtype).expand(self._batch_size, -1, -1)
        self._ac_dist = MultivariateNormal(mean, covariance_matrix=cov)
    
    def _check_bound(self):
        if self._ac_lb is not None or self._ac_ub is not None:
            if self._ac_lb is None or self._ac_ub is None:
                return False
            self._ac_lb = self._ac_lb.to(device=self._device)
            self._ac_ub = self._ac_ub.to(device=self._device)
        return True

    def _clamp_ac_samples(self, ac):
        if self._ac_ub:
            ac = torch.clamp(ac, min=self._ac_lb, max=self._ac_ub)
        return ac

    def _slice_current_step(self, t):
        return slice(t * self._ac_dim, (t + 1) * self._ac_dim)

    def _flatten_batch_dim(self, tensor):
        return torch.flatten(tensor, end_dim=1)

    def _recover_batch_dim(self, tensor):
        return tensor.view((self._batch_size, self._num_samples) + tensor.shape[1:])

    def _evaluate_trajectories(self, obs, ac):
        obs = obs.unsqueeze(1).expand(-1, self._num_samples, -1)  
        # shape = (batch_size, num_samples, ac_dim)
        obs = self._flatten_batch_dim(obs)
        ac = self._flatten_batch_dim(ac)

        if self._horizon == 1:
            cost_total = torch.squeeze(self._cost_fn(obs, ac))
        else:
            cost_total = torch.zeros(self._batch_size * self._num_samples, 
                device=self._device, dtype=self._dtype)
            for t in range(self._horizon):
                ac_t = ac[:, self._slice_current_step(t)]
                obs = self._dyn(obs, ac_t)
                cost_total += self._cost_fn(obs, ac_t)
            if self._terminal_obs_cost:
                cost_total += self._terminal_obs_cost(obs)
        cost_total = self._recover_batch_dim(cost_total)
        return cost_total

    def _step(self, obs):
        # sample K action trajectories
        ac_samples = self._ac_dist.sample((self._num_samples,))  
        # shape(num_samples, batch_size, ac_dim)
        ac_samples = ac_samples.transpose(0, 1)  
        # shape(batch_size, num_samples, ac_dim)
        ac_samples = self._clamp_ac_samples(ac_samples)

        cost_total = self._evaluate_trajectories(obs, ac_samples)  
        _, topk_idx = torch.topk(cost_total, 
            self._num_elite, dim=1, largest=False, sorted=False)
        topk_idx = topk_idx.unsqueeze(2).expand(-1, -1, self._horizon * self._ac_dim)
        top_samples = ac_samples.gather(1, topk_idx)

        mean = torch.mean(top_samples, dim=1)  
        cov = batch_cov(top_samples)  
        cov_rk = torch.matrix_rank(cov) < self._horizon * self._ac_dim
        cov_rk = cov_rk.view(-1, 1, 1)
        cov += self._cov_reg * cov_rk

        self._ac_dist = MultivariateNormal(mean, covariance_matrix=cov)

        return top_samples

    def solve(
        self, obs: torch.Tensor, get_log_probs: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Do the CEM to solve for best actions.

        Args: 
            obs: shape(obs_dim, ) or (batch_size, obs_dim) the current observation.
            get_log_probs: whether or not to return the log probabilities of the 
                returned actions.
        Returns:
            ac_soln: the action solution. Only the action at the FIRST time step of
                the trajectory is returned.
            log_probs: the log probabilities of the returned actions.
        """
        obs = obs.to(self._device)
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        assert len(obs.shape) == 2, OBS_SHAPE_ERROR
        # obs.shape == (batch_size, obs_dim)
        self._batch_size = obs.shape[0]
        self._init_ac_dist()

        for _ in range(self._num_iter):
            self._step(obs)
        ac_soln = self._ac_dist.sample((1,))[0]

        if get_log_probs:
            log_probs = self._ac_dist.log_prob(ac_soln).unsqueeze(1)
            return ac_soln[:, :self._ac_dim], log_probs
        return ac_soln[:, :self._ac_dim]
