

import numpy as np
from quintic_ou import QuinticOU


eps = 1 / 52
H = 0
eps = 0.2
H = 0.2
a_k = [0.01,1,0.214, 0.227]
t_array_nodes = [0, 0.03, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 12 / 12, 24 / 12]
fv_nodes = list(np.ones_like(t_array_nodes) * 0.02)  # fix fwd variance at 0.02
#fv_nodes = np.array([0.012,0.018,0.035,0.026,0.027,0.019,0.025,0.025])

rho = -0.65
S0 = 100

quintic_ou = QuinticOU()
quintic_ou.set_parameters(eps, H, a_k, t_array_nodes, fv_nodes, rho, S0)

num_steps = 400
time_step = (1 / 12) / num_steps
#num_steps = 400 * 12 * 10
time_step = 1 / num_steps
num_sims = 10

tt = quintic_ou.get_dates(num_steps, time_step)
quintic_ou.plot_trajectories(num_steps, time_step, num_sims)