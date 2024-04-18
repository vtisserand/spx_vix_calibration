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

```python
from src.models import qHeston
from src.models.kernels import KernelFlavour

model = qHeston(kernel=KernelFlavour.ROUGH)
# To change from the default set of parameters:
model.set_parameters(a=0.21, b=0.08, c=0.0024, H=0.08, eta=1., eps=1/52, rho=-1., fvc=0.3)

S, V = model.generate_paths(n_steps=int(12*6.5*252), length=5, n_sims=100)
```

```mermaid
graph LR
    style Pricing fill:#FF5733,stroke:#000000
    style Calibration fill:#98FB98,stroke:#000000
    style Simulating fill:#87CEFA,stroke:#000000

    B["qHeston model <br> (with different kernels)"] -->|Rough| G["Euler scheme for (S,V) sample paths"]
    B -->|Path-dependent| G
    B -->|One factor| G
    B -->|Two factor| G
    
    subgraph Simulating
        direction TB
        style Simulating fill:#bae1ff,stroke:#000000
        G -->|V sample paths| H[VIX levels]

    end

    subgraph Data
        style Data fill:#ffffba,stroke:#000000
        OptionChain[Options data]
        futData[VIX futures data]

    end

    subgraph Pricing
        direction TB
        style Pricing fill:#ffb3ba ,stroke:#000000
        OptionChain --> SPX[SPX market surface]
        OptionChain --> VIX[VIX market surface]
        futData --> fut[VIX market curve]
        G -->|S sample paths| SPX_mod[SPX model surface]
        H -->|VIX sample paths| VIX_mod[VIX model surface]
        H -->|VIX sample paths| VIX_fut[VIX future model curve]
        
        
    end
    
    subgraph Calibration
        direction TB
        style Calibration fill:	#baffc9 ,stroke:#000000
        SPX --> spx_err["SPX error <br> (potentially weighted)"]
        SPX_mod --> spx_err
        VIX --> vix_err["VIX error <br> (potentially weighted)"]
        VIX_mod --> vix_err
        fut --> fut_err[Futures error]
        VIX_fut --> fut_err[Futures error]


        spx_err --> err[Error]
        vix_err --> err
        fut_err --> err

        err -->|Minimization algorithm <br> Nelder-Mead| params[Optimal parameters]
    end
```