"""
Bayesian Online Changepoint Detection (BOCPD) implementation.

Based on Adams & MacKay (2007) with constant hazard rate and Gaussian
observation model using Normal-Inverse-Gamma conjugate prior.

Adapted from regime_aware_portfolio_allocator — standalone, no external deps.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy.special import gammaln

from models.bocpd_regime.config import BOCPDConfig


class BOCPDModel:
    """
    Bayesian Online Changepoint Detection with Gaussian observation model.

    Uses the Normal-Inverse-Gamma conjugate prior for online inference
    of both the mean and variance of the underlying process.
    """

    def __init__(self, config: BOCPDConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset the model state for a new sequence."""
        self.run_length_probs = np.array([1.0])
        self.run_length_values = np.array([0], dtype=np.int64)
        self.posterior_params = np.array([[
            self.config.mu0,
            self.config.kappa0,
            self.config.alpha0,
            self.config.beta0,
        ]])
        self.t = 0
        self.observations = []

    def _hazard(self, r: np.ndarray, hazard_override: Optional[float] = None) -> np.ndarray:
        hazard_value = self.config.hazard if hazard_override is None else hazard_override
        return np.full_like(r, hazard_value, dtype=np.float64)

    def _student_t_log_pdf(
        self, x: float,
        mu: np.ndarray, kappa: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray,
    ) -> np.ndarray:
        df = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (kappa * alpha))
        z = (x - mu) / scale
        log_pdf = (
            gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * np.log(np.pi * df)
            - np.log(scale)
            - ((df + 1) / 2) * np.log(1 + z**2 / df)
        )
        return log_pdf

    def _update_posterior(
        self, x: float,
        mu: np.ndarray, kappa: np.ndarray,
        alpha: np.ndarray, beta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kappa_new = kappa + 1
        mu_new = (kappa * mu + x) / kappa_new
        alpha_new = alpha + 0.5
        beta_new = beta + kappa * (x - mu)**2 / (2 * kappa_new)
        return mu_new, kappa_new, alpha_new, beta_new

    def update(self, x: float, hazard_override: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Process a new observation and update the run length distribution.

        Returns:
            (cp_prob, erl, run_mean, run_var)
        """
        self.t += 1
        self.observations.append(x)

        n_hyp = len(self.run_length_probs)
        mu = self.posterior_params[:, 0]
        kappa = self.posterior_params[:, 1]
        alpha = self.posterior_params[:, 2]
        beta = self.posterior_params[:, 3]

        log_pred_probs = self._student_t_log_pdf(x, mu, kappa, alpha, beta)
        pred_probs = np.exp(log_pred_probs - np.max(log_pred_probs))

        H = self._hazard(self.run_length_values, hazard_override=hazard_override)

        growth_probs = self.run_length_probs * pred_probs * (1 - H)
        cp_prob_unnorm = np.sum(self.run_length_probs * pred_probs * H)

        new_probs = np.zeros(n_hyp + 1)
        new_probs[0] = cp_prob_unnorm
        new_probs[1:] = growth_probs

        total = np.sum(new_probs)
        if total > 0:
            new_probs /= total
        else:
            new_probs = np.zeros_like(new_probs)
            new_probs[0] = 1.0

        self.run_length_probs = new_probs
        self.run_length_values = np.concatenate([
            np.array([0], dtype=np.int64),
            self.run_length_values + 1,
        ])

        mu_new, kappa_new, alpha_new, beta_new = self._update_posterior(
            x, mu, kappa, alpha, beta
        )
        new_posterior = np.zeros((n_hyp + 1, 4))
        new_posterior[0] = [
            self.config.mu0, self.config.kappa0,
            self.config.alpha0, self.config.beta0,
        ]
        new_posterior[1:, 0] = mu_new
        new_posterior[1:, 1] = kappa_new
        new_posterior[1:, 2] = alpha_new
        new_posterior[1:, 3] = beta_new
        self.posterior_params = new_posterior

        erl = np.sum(self.run_length_values * self.run_length_probs)

        map_idx = np.argmax(self.run_length_probs)
        run_mean = self.posterior_params[map_idx, 0]
        map_alpha = self.posterior_params[map_idx, 2]
        map_beta = self.posterior_params[map_idx, 3]
        run_var = map_beta / (map_alpha - 1) if map_alpha > 1 else np.inf

        cp_prob = self.run_length_probs[0]
        return cp_prob, erl, run_mean, run_var

    def get_map_run_length(self) -> int:
        map_idx = int(np.argmax(self.run_length_probs))
        return int(self.run_length_values[map_idx])


def run_bocpd(
    x: pd.Series,
    config: BOCPDConfig,
    max_run_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run BOCPD on a 1D time series.

    Args:
        x: Time series (e.g., 21-day rolling returns) indexed by date
        config: BOCPDConfig with algorithm parameters
        max_run_length: Optional maximum run length to track (memory efficiency)

    Returns:
        DataFrame indexed by date with columns:
        cp_prob, erl, run_mean, run_var
    """
    x_clean = x.dropna()

    # Optional: volatility-scale input to stabilise variance
    if config.use_volatility_scaling:
        rolling_vol = x_clean.rolling(window=config.vol_scale_window).std()
        rolling_vol = rolling_vol.clip(lower=config.vol_scale_floor)
        x_clean = x_clean / rolling_vol
        # Drop NaN introduced by rolling window — feeding NaN to the model
        # corrupts posterior params and causes ERL to collapse to 0 permanently
        x_clean = x_clean.dropna()

    if len(x_clean) == 0:
        raise ValueError("Input series has no valid data after dropping NaN")

    model = BOCPDModel(config)

    results = {"cp_prob": [], "erl": [], "run_mean": [], "run_var": []}
    dates = []

    # Optional: dynamic hazard based on rolling volatility
    hazard_series = None
    if config.use_dynamic_hazard:
        vol = x_clean.rolling(window=config.hazard_vol_window).std()
        vol_mean = vol.rolling(window=config.hazard_vol_z_window).mean()
        vol_std = vol.rolling(window=config.hazard_vol_z_window).std()
        vol_z = (vol - vol_mean) / vol_std
        vol_z = vol_z.clip(-3, 3) * config.hazard_vol_scale

        if config.hazard_mapping == "linear":
            scaled = (vol_z + 3) / 6
            hazard_series = config.hazard_min + (config.hazard_max - config.hazard_min) * scaled
        elif config.hazard_mapping == "tanh":
            scaled = (np.tanh(vol_z) + 1) / 2
            hazard_series = config.hazard_min + (config.hazard_max - config.hazard_min) * scaled
        else:  # sigmoid
            hazard_series = config.hazard_min + (config.hazard_max - config.hazard_min) * (
                1 / (1 + np.exp(-vol_z))
            )

    for date, value in x_clean.items():
        hazard_override = None
        if hazard_series is not None:
            hazard_override = hazard_series.loc[date]
            if pd.isna(hazard_override):
                hazard_override = config.hazard

        cp_prob, erl, run_mean, run_var = model.update(value, hazard_override=hazard_override)
        results["cp_prob"].append(cp_prob)
        results["erl"].append(erl)
        results["run_mean"].append(run_mean)
        results["run_var"].append(run_var)
        dates.append(date)

        # Truncate run length distribution for memory efficiency
        if max_run_length is not None and len(model.run_length_probs) > max_run_length:
            probs = model.run_length_probs
            params = model.posterior_params
            rvals = model.run_length_values

            top_idx = np.argsort(probs)[-max_run_length:]
            top_idx.sort()

            probs = probs[top_idx]
            probs = probs / probs.sum()
            params = params[top_idx]
            rvals = rvals[top_idx]

            model.run_length_probs = probs
            model.posterior_params = params
            model.run_length_values = rvals

    result_df = pd.DataFrame(results, index=pd.DatetimeIndex(dates))
    result_df.index.name = "date"
    result_df = result_df.reindex(x.index)
    return result_df
