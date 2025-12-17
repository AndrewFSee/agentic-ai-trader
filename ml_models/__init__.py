"""
ML Models Package for Short-Term Return Prediction

This package implements a supervised learning pipeline for predicting
5-day and 10-day stock returns using multiple models:
- Logistic Regression (baseline)
- Decision Tree (baseline)
- Random Forest
- XGBoost

Based on Lopez de Prado's "Advances in Financial Machine Learning" and
Jansen's "Machine Learning for Algorithmic Trading".
"""

__version__ = "1.0.0"
__author__ = "Agentic AI Trader"

from .data_collection import collect_all_stocks, fetch_daily_bars
from .feature_engineering import engineer_features, get_feature_columns
from .train_models import train_all_models
from .backtest import backtest_all_models
from .run_pipeline import run_full_pipeline

__all__ = [
    'collect_all_stocks',
    'fetch_daily_bars',
    'engineer_features',
    'get_feature_columns',
    'train_all_models',
    'backtest_all_models',
    'run_full_pipeline'
]
