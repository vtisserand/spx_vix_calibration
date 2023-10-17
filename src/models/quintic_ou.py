import numpy as np
import scipy
from scipy import interpolate
import matplotlib.pyplot as plt

from base_model import BaseModel


class QuinticOU(BaseModel):

    def __init__(self):
        self.eps = None
        self.H = None
        self.eta_tild = None
        self.kappa_tild = None
        self.a_k = None
        self.t_array_nodes = None
        self.fv_nodes = None
        self.spine_order = None


    def set_parameters(self, eps: float, H: float, a_k: list, t_array_nodes: list, fv_nodes: list, spine_order: int = 3):
        self.eps = eps
        self.H = H
        self.eta_tild = self.eps**(self.H-0.5)
        self.kappa_tild = (0.5-self.H)/self.eps
        a_0, a_1, a_3, a_5 = a_k
        self.a_k = np.array([a_0, a_1, 0, a_3, 0, a_5])
        self.t_array_nodes = np.array(t_array_nodes)
        self.fv_nodes = np.array(fv_nodes)
        self.spine_order = spine_order

    def fit(self, data):
        pass

    @staticmethod
    def get_dates(num_steps: int, time_step: float):
        dt = time_step
        T = dt*num_steps
        tt = np.linspace(0., T, num_steps + 1)
        return tt

    @staticmethod
    def compute_polynomial(poly, n, x):
        result = poly[0].reshape(-1, 1)
        for i in range(1, n):
            result = result * x + poly[i].reshape(-1, 1)
        return result

    def generate_ou(self, num_steps: int, tt: np.ndarray, num_sims: int = 1):
        """
        Generate an Ornstein-Uhlenbeck trajectory according to the following scheme:
        """
        # Brownian path
        w1_orig = np.random.normal(0, 1, (num_steps, num_sims))
        w1 = np.concatenate(
            (np.zeros(w1_orig.shape[1])[np.newaxis, :], w1_orig))

        exp1 = np.exp(self.kappa_tild*tt)
        exp2 = np.exp(2*self.kappa_tild*tt)

        diff_exp2 = np.concatenate((np.array([0.]), np.diff(exp2)))
        # to be broadcasted columnwise
        std_vec = np.sqrt(diff_exp2/(2*self.kappa_tild))[:, np.newaxis]
        exp1 = exp1[:, np.newaxis]
        X = (1/exp1)*(self.eta_tild*np.cumsum(std_vec*w1, axis=0))
        Xt = np.array(X[:-1])
        return Xt

    def generate_trajectory(self, tt: np.ndarray, Xt: np.ndarray):
        """
        Generates volatility trajectories according to the quintic OU model.
        """
        std_X_t = np.sqrt(self.eta_tild ** 2 / (2 * self.kappa_tild) * (1 - np.exp(-2 * self.kappa_tild * tt)))
        n = len(self.a_k)

        cauchy_product = np.convolve(self.a_k, self.a_k)
        normal_var = np.sum(
            cauchy_product[np.arange(0, 2 * n, 2)].reshape(-1, 1)
            * std_X_t ** (np.arange(0, 2 * n, 2).reshape(-1, 1))
            * scipy.special.factorial2(np.arange(0, 2 * n, 2).reshape(-1, 1) - 1), axis=0)
        normal_var = normal_var[1:].reshape(-1,1)

        polynomial = self.compute_polynomial(self.a_k[::-1], len(self.a_k), Xt)
        fv_var_curve_spline_sqrt = interpolate.splrep(self.t_array_nodes, np.sqrt(self.fv_nodes), k=self.spine_order)
        fv_curve = (interpolate.splev(tt, fv_var_curve_spline_sqrt, der=0)) ** 2
        fv_curve = fv_curve[:-1].reshape(-1,1)
        volatility = np.sqrt(fv_curve) * (polynomial / np.sqrt(normal_var))
        return volatility

    def plot_trajectories(self, num_steps: int, time_step: int, num_sims: int = 1):
        tt = self.get_dates(num_steps, time_step)
        Xt = self.generate_ou(num_steps, tt, num_sims)
        volatility = self.generate_trajectory(tt, Xt)
        tt = tt[:-1]

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(9, 6))
        for i in range(num_sims):
            axs[0].plot(tt, Xt[:, i])
        plt.sca(axs[0])
        axs[0].set_ylabel("OU Simulations")
        axs[0].set_title(
            "Simulated OU Processes (eps={}, H={}, num_sims={})".format(np.round(self.eps, decimals=3), self.H, num_sims))
        for i in range(num_sims):
            axs[1].plot(tt, volatility[:, i])
        plt.sca(axs[0])
        axs[1].set_ylabel("Volatility")
        axs[1].set_title("Simulated Volatility Paths (num_sims={})".format(num_sims))

        plt.subplots_adjust(top=0.78)
        fig.suptitle("Quintic OU Model Simulations", fontsize=22, y=0.943)
        fig.subplots_adjust(hspace=0.4)
        # plt.savefig("Quintic_OU_simulations.pdf")
        plt.show()




