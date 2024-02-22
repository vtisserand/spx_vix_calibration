import numpy as np

from src.models.base_model import BaseModel


class rBergomi(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The rough Bergomi model...
        """
        super().__init__(initial_price)
        self.set_parameters()

    def __repr__(self):
        dynamics = "This is an instance of a rough Bergomi model."
        params = f"It has parameters vol_init: {self.vol_init}, H: {self.H}, eta: {self.eta}, eps: {self.eps}, rho: {self.rho}, fvc: {self.fvc}"
        return dynamics + '\n' + params

    def set_parameters(
        self,
        vol_init: float=0.25,
        H: float=0.2,
        eta: float=0.7,
        eps: float=1/52,
        rho: float=-0.6,
        fvc: float=0.35
    ):
        """
        Set of coherent dummy parameters to play around without market data, from the original rough Heston paper.
        """
        (
            self.vol_init,
            self.H,
            self.eta,
            self.eps,
            self.rho,
            self.fvc
        ) = vol_init, H, eta, eps, rho, fvc


    def fit(
        self,
        strikes: np.ndarray | list[float],
        prices: np.ndarray | list[float],
        forward_price: float = 100,
        maturity: float = 1,
    ):
        """
        """
        pass

    def generate_paths(self, n_steps: int, length: int, n_sims: int=1):
        # Uncorrelated brownians
        w1, w2 = np.random.normal(0, 1, (n_steps*length, n_sims)), np.random.normal(0, 1, (n_steps*length, n_sims))

        tt = np.linspace(0,n_steps*length,n_steps*length+1)
        dt = 1/n_steps
        
        # Euler 3 scheme for variance process
        Xt = np.zeros(n_sims).reshape(1,-1)
        for j in range(n_steps*length):
            tj = tt[j+1]
            ti_s = tt[:j+2]
            std_ji = np.sqrt(((tj-ti_s[:-1])**(2*self.H)-(tj-ti_s[1:])**(2*self.H))/(2*self.H))
            temp = np.sum(std_ji.reshape(-1,1)*w1[:j+1],axis=0)
            Xt=np.append(Xt,temp.reshape(1,-1),axis=0)

        drift_rbergomi = -0.25*self.eta**2*tt**(2*self.H)
        volatility = np.exp(0.5*self.eta*np.sqrt(2*self.H)*Xt+drift_rbergomi.reshape(-1,1))
        volatility = np.sqrt(self.fvc)*volatility

        log_S = np.ones(n_sims).reshape(1,-1)*np.log(self.initial_price)
        for j in range(n_steps*length):
            log_S_next = log_S[j]-0.5*volatility[j]**2*dt+volatility[j]*np.sqrt(dt)*\
                (self.rho*w1[j]+np.sqrt(1-self.rho**2)*w2[j])
            log_S=np.append(log_S,log_S_next.reshape(1,-1),axis=0)

        return np.exp(log_S)

