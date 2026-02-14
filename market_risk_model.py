"""
Market Risk Model — Calibrated Drawdown Probability + Forward Volatility

Replaces VIX ROC binary risk signal with continuous risk probabilities.
Uses VIX-based universal features that work across all equities.

Architecture (validated via walk-forward comparison of 9 approaches):
    1. P(DD > 3% in 10 days) = IsotonicRegression(vix_percentile_252d)
       - AUC=0.651, Brier=0.121 — BEATS all ML approaches tested
    2. Forward realized vol = HistGradientBoostingRegressor(12 features)
       - R²≈0.17 — useful for position sizing and stop-loss guidance

Why not ML for drawdown probability?
    Exhaustive walk-forward testing showed single-feature isotonic calibration
    (AUC=0.651) beats: GBM classifier (0.530), HistGBR DD regression (0.576),
    HistGBR vol→isotonic (0.591), Logistic L1 (0.500), Logistic Ridge (0.643),
    multi-isotonic ensembles (0.629-0.635), and vol+vixpct blend (0.633).
    Root cause: 16.8% positive rate + correlated features → all ML overfits.
    Occam's razor wins decisively.

Data:  SPY + VIX from yfinance (trained on-the-fly, cached daily)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score, brier_score_loss, mean_squared_error, r2_score
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# 12 features — removed correlated/noisy features from the original 21:
#   DROPPED: vix_zscore_20d (corr with 60d), vix_roc_2d (AUC=0.526, noise),
#   vix_roc_10d (redundant), vix_acceleration (AUC=0.504, noise),
#   rv_10d (r=0.90 with rv_5d AND rv_20d), rv_60d (corr with rv_20d + vix),
#   vol_ratio_10_60 (redundant), spy_ret_20d (r=0.81 with trailing_dd_20d),
#   trailing_dd_20d (r=0.90 with trailing_dd_10d)
FEATURE_NAMES = [
    # VIX features (6)
    "vix_level",             # AUC=0.662 — core VIX state
    "vix_zscore_60d",        # AUC=0.640 — relative VIX vs long history
    "vix_roc_5d",            # AUC=0.549 — short-term VIX momentum
    "vix_roc_20d",           # AUC=0.614 — medium-term VIX momentum
    "vix_percentile_252d",   # AUC=0.670 — highest individual AUC
    "vol_of_vix_20d",        # AUC=0.569 — vol-of-vol
    # Realized vol features (3)
    "rv_5d",                 # AUC=0.637 — short-term realized vol
    "rv_20d",                # AUC=0.631 — medium-term realized vol
    "vol_ratio_5_20",        # AUC=0.565 — vol term structure
    # Variance risk premium (1)
    "vrp",                   # AUC=0.551 — implied vs realized
    # Price features (2)
    "spy_ret_5d",            # AUC=0.555 — recent price momentum
    "trailing_dd_10d",       # AUC=0.617 — recent drawdown (clustering)
]


def compute_features(spy_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    """
    Compute feature matrix from SPY and VIX close prices.
    All features use data up to time *t* only — no look-ahead bias.
    """
    df = pd.DataFrame(index=spy_close.index)

    vix = vix_close.reindex(spy_close.index, method="ffill")
    spy_ret = spy_close.pct_change()

    # ── VIX features ────────────────────────────────────────────
    df["vix_level"] = vix

    ma60 = vix.rolling(60, min_periods=60).mean()
    std60 = vix.rolling(60, min_periods=60).std()
    df["vix_zscore_60d"] = (vix - ma60) / std60

    for w in [5, 20]:
        df[f"vix_roc_{w}d"] = vix.pct_change(w)

    df["vix_percentile_252d"] = vix.rolling(252, min_periods=252).apply(
        lambda x: (x[-1] > x[:-1]).mean(), raw=True
    )

    vix_ret = vix.pct_change()
    df["vol_of_vix_20d"] = vix_ret.rolling(20, min_periods=20).std()

    # ── Realised vol features ───────────────────────────────────
    for w in [5, 20]:
        df[f"rv_{w}d"] = spy_ret.rolling(w, min_periods=w).std() * np.sqrt(252)

    df["vol_ratio_5_20"] = df["rv_5d"] / df["rv_20d"].replace(0, np.nan)

    # ── Variance risk premium ───────────────────────────────────
    df["vrp"] = (vix / 100) / df["rv_20d"].replace(0, np.nan)

    # ── Price features ──────────────────────────────────────────
    df["spy_ret_5d"] = spy_close.pct_change(5)

    rolling_max_10 = spy_close.rolling(10, min_periods=10).max()
    df["trailing_dd_10d"] = (spy_close - rolling_max_10) / rolling_max_10

    return df[FEATURE_NAMES]


def compute_targets(
    spy_close: pd.Series,
    drawdown_threshold: float = 0.03,
    forward_window: int = 10,
) -> pd.DataFrame:
    """
    Compute forward-looking targets.
    WARNING — these use FUTURE data. For training only, never live.
    """
    close_vals = spy_close.values
    ret_vals = spy_close.pct_change().values
    n = len(close_vals)

    fwd_max_dd = np.full(n, np.nan)
    fwd_vol = np.full(n, np.nan)

    for i in range(n - forward_window):
        future = close_vals[i + 1 : i + 1 + forward_window]
        entry_price = close_vals[i]
        max_drop = np.min(future / entry_price - 1)
        fwd_max_dd[i] = -max_drop  # positive = drawdown magnitude

        fwd_ret = ret_vals[i + 1 : i + 1 + forward_window]
        fwd_vol[i] = np.nanstd(fwd_ret) * np.sqrt(252)

    targets = pd.DataFrame(index=spy_close.index)
    targets["fwd_max_dd"] = fwd_max_dd
    targets["drawdown_flag"] = (fwd_max_dd > drawdown_threshold).astype(float)
    targets["fwd_realized_vol"] = fwd_vol
    return targets


# =============================================================================
# MODEL
# =============================================================================

class MarketRiskModel:
    """
    Market risk model: calibrated drawdown probability + forward vol prediction.

    1. P(DD > 3%) = IsotonicRegression(vix_percentile_252d → drawdown_flag)
       - AUC=0.651 walk-forward — beats all ML approaches tested
       - Simple, robust, well-calibrated probabilities
    2. Forward vol = HistGradientBoostingRegressor(12 features → fwd_realized_vol)
       - R²≈0.17 — useful for position sizing guidance
    """

    # Hyperparams for vol regressor only
    _HGBR_PARAMS = dict(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=40,
        max_bins=128,
        l2_regularization=1.0,
        random_state=42,
    )

    def __init__(
        self,
        drawdown_threshold: float = 0.03,
        forward_window: int = 10,
        data_years: int = 15,
    ):
        self.drawdown_threshold = drawdown_threshold
        self.forward_window = forward_window
        self.data_years = data_years

        # DD probability: isotonic calibration on vix_percentile_252d
        self._dd_calibrator: Optional[IsotonicRegression] = None
        # DD magnitude estimate: isotonic on vix_percentile → mean DD
        self._dd_magnitude_calibrator: Optional[IsotonicRegression] = None
        # Vol prediction: HistGBR on 12 features
        self._vol_regressor: Optional[HistGradientBoostingRegressor] = None
        self._trained = False
        self._metrics: Dict = {}
        self._vol_feature_importance: Dict[str, float] = {}

        # Data cache
        self._spy_close: Optional[pd.Series] = None
        self._vix_close: Optional[pd.Series] = None
        self._features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _download_data(self) -> Tuple[pd.Series, pd.Series]:
        """Download SPY + VIX history from yfinance."""
        if not _YF_AVAILABLE:
            raise ImportError("yfinance is required but not installed")

        end = dt.date.today()
        start = end - dt.timedelta(days=self.data_years * 365 + 90)

        spy = yf.download("SPY", start=str(start), end=str(end), progress=False)
        vix = yf.download("^VIX", start=str(start), end=str(end), progress=False)

        for frame in [spy, vix]:
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(0)

        return spy["Close"], vix["Close"]

    # ------------------------------------------------------------------
    # Walk-forward validation
    # ------------------------------------------------------------------
    def _validate(self, combined: pd.DataFrame) -> None:
        """
        Expanding-window walk-forward validation (last 4 years as test).

        For each fold:
          - Fit IsotonicRegression: vix_percentile_252d → P(DD_flag)
          - Fit HistGBR: 12 features → forward realized vol
          - Evaluate: AUC, Brier (DD prob), Vol R²
        """
        years = sorted(combined.index.year.unique())
        if len(years) < 5:
            self._metrics = {"note": "Insufficient history for walk-forward validation"}
            return

        test_years = years[-4:]

        all_y_flag, all_p_prob = [], []
        all_y_vol, all_p_vol = [], []
        folds = []

        vp_idx = FEATURE_NAMES.index("vix_percentile_252d")

        for test_year in test_years:
            train = combined[combined.index.year < test_year]
            test = combined[combined.index.year == test_year]
            if len(train) < 500 or len(test) < 50:
                continue

            X_tr = train[FEATURE_NAMES].values
            X_te = test[FEATURE_NAMES].values
            y_tr_flag = train["drawdown_flag"].values
            y_te_flag = test["drawdown_flag"].values
            y_tr_vol = train["fwd_realized_vol"].values
            y_te_vol = test["fwd_realized_vol"].values

            # DD probability: vix_percentile → isotonic → P(DD)
            calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            calibrator.fit(X_tr[:, vp_idx], y_tr_flag)
            probs = calibrator.predict(X_te[:, vp_idx])

            # Vol regression
            vol_reg = HistGradientBoostingRegressor(**self._HGBR_PARAMS)
            vol_reg.fit(X_tr, y_tr_vol)
            pred_vol = vol_reg.predict(X_te)

            all_y_flag.extend(y_te_flag)
            all_p_prob.extend(probs)
            all_y_vol.extend(y_te_vol)
            all_p_vol.extend(pred_vol)

            try:
                auc = roc_auc_score(y_te_flag, probs)
            except Exception:
                auc = float("nan")

            vol_r2 = r2_score(y_te_vol, pred_vol)

            folds.append({
                "year": int(test_year),
                "auc_roc": round(auc, 3),
                "brier": round(brier_score_loss(y_te_flag, probs), 4),
                "vol_rmse": round(float(np.sqrt(mean_squared_error(y_te_vol, pred_vol))), 4),
                "vol_r2": round(float(vol_r2), 3),
                "dd_base_rate": round(float(np.mean(y_te_flag)), 3),
                "n_test": len(y_te_flag),
            })

        if not all_y_flag:
            self._metrics = {"note": "Walk-forward produced no valid folds"}
            return

        try:
            overall_auc = roc_auc_score(all_y_flag, all_p_prob)
        except Exception:
            overall_auc = float("nan")

        overall_vol_r2 = r2_score(all_y_vol, all_p_vol)

        self._metrics = {
            "overall_auc_roc": round(overall_auc, 3),
            "overall_brier": round(brier_score_loss(all_y_flag, all_p_prob), 4),
            "overall_vol_rmse": round(float(np.sqrt(mean_squared_error(all_y_vol, all_p_vol))), 4),
            "overall_vol_r2": round(float(overall_vol_r2), 3),
            "dd_base_rate": round(float(np.mean(all_y_flag)), 3),
            "n_total_test": len(all_y_flag),
            "folds": folds,
        }

    # ------------------------------------------------------------------
    # Train + predict
    # ------------------------------------------------------------------
    def train_and_predict(self) -> Dict:
        """
        Full pipeline: download -> features -> validate -> train -> predict today.
        """
        if not _SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        # Download
        spy_close, vix_close = self._download_data()
        self._spy_close = spy_close
        self._vix_close = vix_close

        # Features + targets
        features = compute_features(spy_close, vix_close)
        targets = compute_targets(spy_close, self.drawdown_threshold, self.forward_window)
        combined = pd.concat([features, targets], axis=1).dropna()

        X = combined[FEATURE_NAMES].values
        y_dd_flag = combined["drawdown_flag"].values
        y_dd = combined["fwd_max_dd"].values
        y_vol = combined["fwd_realized_vol"].values
        vp_idx = FEATURE_NAMES.index("vix_percentile_252d")

        print(f"  [MarketRisk] Training on {len(X)} samples, "
              f"{len(FEATURE_NAMES)} features, "
              f"DD base rate {y_dd_flag.mean():.1%}")

        # Walk-forward validation
        self._validate(combined)
        auc = self._metrics.get("overall_auc_roc")
        vol_r2 = self._metrics.get("overall_vol_r2")
        if auc is not None:
            print(f"  [MarketRisk] Walk-forward AUC={auc:.3f}, Vol-R2={vol_r2:.3f}")

        # Final training on all data
        # 1. DD probability calibrator: vix_percentile → P(DD_flag)
        self._dd_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._dd_calibrator.fit(X[:, vp_idx], y_dd_flag)

        # 2. DD magnitude calibrator: vix_percentile → expected DD magnitude
        self._dd_magnitude_calibrator = IsotonicRegression(out_of_bounds="clip")
        self._dd_magnitude_calibrator.fit(X[:, vp_idx], y_dd)

        # 3. Vol regressor: 12 features → forward vol
        self._vol_regressor = HistGradientBoostingRegressor(**self._HGBR_PARAMS)
        self._vol_regressor.fit(X, y_vol)

        # Feature importance for vol model (correlation-based proxy)
        self._vol_feature_importance = {}
        for i, feat in enumerate(FEATURE_NAMES):
            col = X[:, i]
            self._vol_feature_importance[feat] = abs(float(np.corrcoef(col, y_vol)[0, 1]))

        self._trained = True
        self._features = features

        # Predict current state (latest row of features)
        X_now = features.iloc[[-1]].values
        vp_now = float(X_now[0, vp_idx])
        dd_prob = float(np.clip(self._dd_calibrator.predict([vp_now])[0], 0, 1))
        pred_dd = float(np.clip(self._dd_magnitude_calibrator.predict([vp_now])[0], 0, 1))
        fwd_vol = float(self._vol_regressor.predict(X_now)[0])

        return self._build_assessment(
            dd_prob, pred_dd, fwd_vol,
            features.iloc[-1], spy_close, vix_close
        )

    # ------------------------------------------------------------------
    # Assessment builder
    # ------------------------------------------------------------------
    def _build_assessment(
        self,
        dd_prob: float,
        pred_dd_magnitude: float,
        fwd_vol: float,
        latest: pd.Series,
        spy_close: pd.Series,
        vix_close: pd.Series,
    ) -> Dict:
        """Convert raw predictions into a structured dict for the agent."""

        # Risk level
        if dd_prob < 0.15:
            risk_level = "LOW"
        elif dd_prob < 0.30:
            risk_level = "MODERATE"
        elif dd_prob < 0.50:
            risk_level = "ELEVATED"
        elif dd_prob < 0.70:
            risk_level = "HIGH"
        else:
            risk_level = "EXTREME"

        # Vol trend
        rv_5 = float(latest.get("rv_5d", 0))
        rv_20 = float(latest.get("rv_20d", 0))
        if rv_5 > rv_20 * 1.2:
            vol_trend = "RISING"
        elif rv_5 < rv_20 * 0.8:
            vol_trend = "FALLING"
        else:
            vol_trend = "STABLE"

        # Position sizing guidance
        if dd_prob < 0.15:
            pos_pct, stop_mult = 1.0, 1.0
        elif dd_prob < 0.30:
            pos_pct, stop_mult = 0.85, 1.2
        elif dd_prob < 0.50:
            pos_pct, stop_mult = 0.65, 1.4
        elif dd_prob < 0.70:
            pos_pct, stop_mult = 0.40, 1.7
        else:
            pos_pct, stop_mult = 0.20, 2.0

        # Human message
        parts = [
            f"{risk_level} drawdown risk ({dd_prob:.0%}).",
            f"Expected forward DD {pred_dd_magnitude * 100:.2f}%.",
            f"Predicted forward vol {fwd_vol * 100:.1f}%"
            f" vs current {rv_20 * 100:.1f}% ({vol_trend}).",
        ]
        if risk_level in ("HIGH", "EXTREME"):
            parts.append(
                "Consider reducing position size significantly or waiting "
                "for conditions to improve."
            )
        elif risk_level == "ELEVATED":
            parts.append(
                f"Suggest reducing position to {pos_pct:.0%} of normal "
                f"and widening stops {stop_mult}x."
            )

        # Top risk drivers (from vol model + vix_percentile as primary)
        top_drivers = sorted(
            self._vol_feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )[:5]

        return {
            # Drawdown prediction
            "drawdown_probability": round(dd_prob, 3),
            "predicted_max_dd_pct": round(pred_dd_magnitude * 100, 2),
            "drawdown_threshold_pct": round(self.drawdown_threshold * 100, 1),
            "drawdown_window_days": self.forward_window,
            "drawdown_risk_level": risk_level,
            # Forward vol prediction
            "predicted_fwd_vol_pct": round(fwd_vol * 100, 1),
            "current_realized_vol_pct": round(rv_20 * 100, 1),
            "vol_trend": vol_trend,
            # Position sizing guidance
            "suggested_position_pct": round(pos_pct, 2),
            "suggested_stop_multiplier": round(stop_mult, 1),
            # Context
            "current_vix": round(float(vix_close.iloc[-1]), 1),
            "vix_zscore_60d": round(float(latest.get("vix_zscore_60d", 0)), 2),
            "vix_roc_5d_pct": round(float(latest.get("vix_roc_5d", 0)) * 100, 1),
            "variance_risk_premium": round(float(latest.get("vrp", 0)), 2),
            # Model quality
            "validation": self._metrics,
            # Top risk drivers
            "top_risk_drivers": top_drivers,
            "message": " ".join(parts),
        }


# =============================================================================
# LAZY SINGLETON (daily cache)
# =============================================================================

_model_instance: Optional[MarketRiskModel] = None
_cached_assessment: Optional[Dict] = None
_cache_date: Optional[str] = None


def _get_market_risk_assessment(force_refresh: bool = False) -> Dict:
    global _model_instance, _cached_assessment, _cache_date

    today = dt.date.today().isoformat()
    if _cached_assessment is not None and _cache_date == today and not force_refresh:
        return _cached_assessment

    print("[MarketRisk] Training market risk model (first call of the day)...")
    _model_instance = MarketRiskModel()
    _cached_assessment = _model_instance.train_and_predict()
    _cache_date = today
    return _cached_assessment


# =============================================================================
# TOOL FUNCTION
# =============================================================================

def market_risk_tool_fn(state: dict, args: dict) -> dict:
    """
    ML-based market risk assessment.

    Predicts drawdown probability and forward volatility for the agent.
    Replaces VIX ROC binary signal with continuous probabilities.
    """
    if "tool_results" not in state:
        state["tool_results"] = {}

    symbol = args.get("symbol", "SPY").upper()

    try:
        assessment = _get_market_risk_assessment()

        if "error" in assessment:
            state["tool_results"]["market_risk"] = {
                "symbol": symbol,
                "error": assessment["error"],
            }
            return state

        result = dict(assessment)
        result["symbol"] = symbol
        result["note"] = (
            "Risk assessment is MARKET-WIDE (based on SPY/VIX features). "
            "For high-beta stocks, drawdown risk and vol may be amplified. "
            "Use ATR from price data to set actual stop levels."
        )

        state["tool_results"]["market_risk"] = result

    except Exception as e:
        state["tool_results"]["market_risk"] = {
            "symbol": symbol,
            "error": f"Market risk model failed: {e}",
        }

    return state
