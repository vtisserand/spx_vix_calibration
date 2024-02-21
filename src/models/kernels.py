import numpy as np
from scipy.special import gamma

# Kernel broadly used for stochastic Volterra equations (SVEs),
# formalism from http://dx.doi.org/10.2139/ssrn.4684016


class KernelFlavour:
    ROUGH = "rough"
    PATH_DEPENDENT = "path_dependent"
    ONE_FACTOR = "one_factor"
    TWO_FACTOR = "two_factor"

def mittag_leffler(alpha: float, beta: float, z:float, n: int=30):
    """
    Compute the Mittag-Leffler function E_alpha,beta(z). 
    Proxy good enough with first n=30 components, but let's keep this in mind.
    """
    return sum(z**k / gamma(alpha*k + beta) for k in range(n))

def rough_kernel(
    t: float,
    H: float,
    eta: float,
):
    return eta * t ** (H - 1 / 2)


def path_dependent_kernel(
    t: float,
    H: float,
    eta: float,
    eps: float,
):
    return eta * (t + eps) ** (H - 1 / 2)


def one_factor_kernel(
    t: float,
    H: float,
    eta: float,
    eps: float,
):
    return eta * eps ** (H - 1 / 2) * np.exp(-(1 / 2 - H) * (t / eps))


def two_factor_kernel(
    t: float,
    H: float,
    eta: float,
    H_l: float,
    eta_l: float,
    eps: float,
):
    return eta * eps ** (H - 1 / 2) * np.exp(-(1 / 2 - H) * (t / eps)) + \
        eta_l * eps ** (H_l - 1 / 2) * np.exp(-(1 / 2 - H_l) * (t / eps))
