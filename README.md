# Joint calibration of SPX and VIX smiles

We are investigating the performance of a new class of stochastic volatility models for fitting SPX and VIX smiles. In particular, we put the emphasis on rough volatility and how to reconcile such dynamics with classic Markovian models.

![The joint calibration problem](report/img/surfaces.png)

## Structure of the code

There are two big machineries in the code: the models that usually have a method to generate sample paths, and the stylized facts on which we evaluate the generated prices.

```
❯ tree
.
├── __init__.py
├── config.py
├── models
│   ├── __init__.py
│   ├── base_model.py
│   ├── fbm.py
│   ├── heston.py
│   ├── kernels.py
│   ├── qHeston.py
│   ├── quintic_ou.py
│   ├── rBergomi.py
│   ├── rHeston.py
│   ├── sabr.py
│   └── test.py
├── stylized_facts.py
└── utils.py
```

Example use:

```
from src.models import qHeston
from src.models.kernels import KernelFlavour

model = qHeston(kernel=KernelFlavour.ROUGH)
# To change from the default set of parameters:
model.set_parameters(a=0.21, b=0.08, c=0.0024, H=0.08, eta=1., eps=1/52, rho=-1., fvc=0.3)

S, V = model.generate_paths(n_steps=int(12*6.5*252), length=5, n_sims=100)
```