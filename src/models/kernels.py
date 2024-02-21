import numpy as np

# Kernel broadly used for stochastic Volterra equations (SVEs),
# formalism from http://dx.doi.org/10.2139/ssrn.4684016

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
