import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky
from scipy.integrate import quad
from math import exp, log, pi, sqrt

from src.models.base_model import BaseModel


class Heston(BaseModel):
    def __init__(self, initial_price: float = 100):
        """
        The Heston model is a stochastic volatility model that assumes a geometric brownian motion dynamic
        for the stock price and a CIR process for the instantaneous variance:
        
        dS_t &= \mu S_t \, dt + \sqrt{V_t}S_t \, dW_t \\
        dV_t &= \kappa (\theta - v_t) \, dt + \sigma \sqrt{V_t} \, dB_t
        \langle dW_t, dB_t \rangle = \rho dt
        """
        super().__init__(initial_price)
        self.set_parameters()

    def set_parameters(
        self,
        vol_init: float = 0.25,
        mu: float = 0.1,
        kappa: float = 1.1,
        theta: float = 0.12,
        sigma: float = 0.80,
        rho: float = -0.6,
    ):
        """
        Set of coherent dummy parameters to play around without market data.
        """
        (
            self.vol_init,
            self.mu,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        ) = vol_init, mu, kappa, theta, sigma, rho

    def fit(
        self,
        strikes: np.ndarray | list[float],
        prices: np.ndarray | list[float],
        forward_price: float = 100,
        maturity: float = 1,
    ):
        """
        Fits a Heston model on a surface slice by minimizing the MSE between market prices and model prices.
        """

        # Objective function to minimize
        def objective_function(parameters):
            self.set_parameters(*parameters)
            model_prices = [
                self.calculate_price(strike, forward_price, maturity)
                for strike in strikes
            ]
            return np.mean((np.array(model_prices) - np.array(prices)) ** 2)

        # Initial guess for parameters
        initial_guess = [
            self.vol_init,
            self.mu,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
        ]

        # Minimize the objective function
        result = minimize(objective_function, initial_guess, method="Nelder-Mead")

        if result.success:
            self.set_parameters(*result.x)
        else:
            raise ValueError(f"Fitting failed: {result.message}")

    def calculate_price(self, strike, forward_price, maturity, option_type='C'):
        # Characteristic function of the Heston model
        def characteristic_function(u, t):
            kappa, theta, sigma, rho = self.kappa, self.theta, self.sigma, self.rho
            xi = kappa - sigma * rho * u * 1j
            d = sqrt((rho * sigma * u * 1j - xi) ** 2 - sigma ** 2 * (2 * u * 1j - u ** 2))
            g = (xi - rho * sigma * u * 1j - d) / (xi - rho * sigma * u * 1j + d)
            D = (xi - rho * sigma * u * 1j - d) / sigma ** 2 * ((1 - exp(-d * t)) / (1 - g * exp(-d * t)))
            C = kappa * (theta - sigma ** 2 / 2) * ((1 - exp(-d * t)) / (1 - g * exp(-d * t)))
            return exp(C + D * self.vol_init + 1j * u * (log(forward_price) + (self.mu - 0.5 * self.vol_init) * t))

        # Calculate option price based on option type
        if option_type.upper() == 'C':
            # Call option price
            integrand = lambda u: (exp(-1j * log(strike) * u) * characteristic_function(u - 0.5j, maturity) / (u ** 2 + 0.25)).real
            integral_result, _ = quad(integrand, 0, 1000)  # Adjust upper limit as needed
            option_price = forward_price - 0.5 - 1 / pi * exp(-self.mu * maturity) * integral_result
        elif option_type.upper() == 'P':
            # Put option price (use put-call parity)
            call_price = self.calculate_price(strike, forward_price, maturity, option_type='C')
            option_price = call_price + strike * exp(-self.mu * maturity) - forward_price
        else:
            raise ValueError("Invalid option type. Please specify 'C' for call or 'P' for put.")

        return option_price

    def generate_paths(self, n_steps: int, length: int, n_sims: int=1):
        dt = length / n_steps
        cov_matrix = np.array(
            [[dt, dt * self.rho], [dt * self.rho, dt]]
        )  # Covariance matrix
        cholesky_matrix = cholesky(cov_matrix, lower=True)  # Cholesky decomposition

        dw_corr, dz_corr = np.random.normal(0, 1, size=(2, n_steps))
        correlated_random_variables = np.dot(cholesky_matrix, [dw_corr, dz_corr])

        S = np.zeros(n_steps)
        vol = np.zeros_like(S)

        S[0] = self.initial_price
        vol[0] = self.vol_init

        for t in range(1, n_steps):
            vol_sq = np.maximum(vol[t - 1], 0)  # Avoid negative volatilities
            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * vol_sq) * dt
                + np.sqrt(vol_sq) * correlated_random_variables[0, t]
            )
            vol[t] = (
                vol[t - 1]
                + self.kappa * (self.theta - vol[t - 1]) * dt
                + self.sigma * np.sqrt(vol_sq) * correlated_random_variables[1, t]
            )

        return S
