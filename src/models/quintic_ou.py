import numpy as np

from base_model import BaseModel


class QuinticOU(BaseModel):

    def __init__(self):
        pass

    def set_parameters(self, eps: float, H: float):
        self.eps = eps
        self.H = H

    def fit(self, data):
        pass

    def generate_ou(self, num_steps: int, time_step: float, num_sims: int = 1):
        """
        Generate an Ornstein-Uhlenbeck trajectory according to the following scheme:

        """
        # Brownian path

        w1_orig = np.random.normal(0, 1, (num_steps, num_sims))
        w1 = np.concatenate(
            (np.zeros(w1_orig.shape[1])[np.newaxis, :], w1_orig))

        eta_tild = self.eps**(self.H-0.5)
        kappa_tild = (0.5-self.H)/self.eps

        dt = time_step
        T = dt*num_steps
        tt = np.linspace(0., T, num_steps + 1)

        exp1 = np.exp(kappa_tild*tt)
        exp2 = np.exp(2*kappa_tild*tt)

        diff_exp2 = np.concatenate((np.array([0.]), np.diff(exp2)))
        # to be broadcasted columnwise
        std_vec = np.sqrt(diff_exp2/(2*kappa_tild))[:, np.newaxis]
        exp1 = exp1[:, np.newaxis]
        X = (1/exp1)*(eta_tild*np.cumsum(std_vec*w1, axis=0))
        Xt = np.array(X[:-1])
        return Xt

    def generate_trajectory(self, num_steps, time_step):
        pass
