"""
ML Prediction Tool for Trading Agent
Provides multi-model predictions with comprehensive performance context
"""

import os
import sys
import json
import joblib
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Any

# Resolve ml_models directory
try:
    ml_models_path = Path(__file__).parent / "ml_models"
except NameError:
    ml_models_path = Path.cwd() / "ml_models"


def _import_from_ml_models(module_name: str):
    """Import a module from ml_models/ by absolute file path, avoiding config.py collisions."""
    module_file = ml_models_path / f"{module_name}.py"
    if not module_file.exists():
        raise ImportError(f"{module_file} not found")
    spec = importlib.util.spec_from_file_location(module_name, str(module_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _fe_mod = _import_from_ml_models("feature_engineering")
    engineer_features = _fe_mod.engineer_features
    _dc_mod = _import_from_ml_models("data_collection")
    fetch_daily_bars = _dc_mod.fetch_daily_bars
    import pandas as pd
    import numpy as np
    import datetime as dt
    _ML_AVAILABLE = True
except ImportError as e:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = str(e)
except Exception as e:
    _ML_AVAILABLE = False
    _ML_IMPORT_ERROR = str(e)


# Model performance context (from latest tuning results)
MODEL_PERFORMANCE = {
    "3d": {
        "best_model": "Random Forest",
        "models": {
            "Random Forest": {
                "sharpe": 1.517,
                "return": 52.31,
                "win_rate": 17.91,
                "strengths": "Best for short-term predictions, handles non-linear patterns",
                "weaknesses": "Lower win rate, can be volatile"
            },
            "XGBoost": {
                "sharpe": 1.341,
                "return": 53.04,
                "win_rate": 22.34,
                "strengths": "Consistent performance, good win rate",
                "weaknesses": "Slightly lower Sharpe than RF for 3d"
            },
            "Logistic Regression": {
                "sharpe": 1.167,
                "return": 54.35,
                "win_rate": 24.96,
                "strengths": "Highest win rate, interpretable",
                "weaknesses": "Lower Sharpe, linear assumptions"
            },
            "Decision Tree": {
                "sharpe": 0.909,
                "return": 26.48,
                "win_rate": 24.56,
                "strengths": "Simple, interpretable",
                "weaknesses": "Lowest Sharpe, prone to overfitting"
            }
        }
    },
    "5d": {
        "best_model": "XGBoost",
        "models": {
            "XGBoost": {
                "sharpe": 1.343,
                "return": 34.97,
                "win_rate": 19.88,
                "strengths": "Best for medium-term, robust gradient boosting",
                "weaknesses": "Moderate win rate"
            },
            "Logistic Regression": {
                "sharpe": 1.137,
                "return": 35.61,
                "win_rate": 23.61,
                "strengths": "Good win rate, stable",
                "weaknesses": "Lower Sharpe than XGBoost"
            },
            "Random Forest": {
                "sharpe": 1.117,
                "return": 27.91,
                "win_rate": 15.35,
                "strengths": "Ensemble robustness",
                "weaknesses": "Low win rate for 5d, better for 3d"
            },
            "Decision Tree": {
                "sharpe": 0.783,
                "return": 28.02,
                "win_rate": 23.15,
                "strengths": "Interpretable",
                "weaknesses": "Lowest Sharpe"
            }
        }
    },
    "10d": {
        "best_model": "XGBoost",
        "models": {
            "XGBoost": {
                "sharpe": 1.453,
                "return": 36.21,
                "win_rate": 19.48,
                "strengths": "Best for long-term, highest Sharpe for 10d",
                "weaknesses": "Moderate win rate"
            },
            "Logistic Regression": {
                "sharpe": 1.110,
                "return": 22.83,
                "win_rate": 20.25,
                "strengths": "Consistent, interpretable",
                "weaknesses": "Lower Sharpe"
            },
            "Decision Tree": {
                "sharpe": 1.006,
                "return": 22.63,
                "win_rate": 20.30,
                "strengths": "Simple",
                "weaknesses": "Lower performance"
            },
            "Random Forest": {
                "sharpe": 0.915,
                "return": 16.53,
                "win_rate": 13.91,
                "strengths": "Ensemble approach",
                "weaknesses": "Lowest Sharpe for 10d, better for 3d"
            }
        }
    }
}

# Overall pipeline stats
PIPELINE_STATS = {
    "features": 141,
    "feature_evolution": "167 baseline → 196 enhanced → 141 pruned (dropped 55 noisy features)",
    "total_improvement": "+26.0% vs baseline",
    "feature_efficiency": "+49.3% (Sharpe per feature)",
    "training_stocks": 25,
    "training_period": "2020-12-17 to 2025-12-16",
    "validation_method": "60/20/20 train/val/test split",
    "hyperparameter_tuning": "Optuna (50 trials per horizon on AAPL, NVDA, JPM)"
}


def load_model(symbol: str, horizon: int, model_name: str):
    """Load a trained model from disk."""
    model_dir = ml_models_path / "saved_models" / f"{symbol}_{horizon}d"
    
    # Map model names to file names
    model_files = {
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "Logistic Regression": "logistic.pkl",
        "Decision Tree": "decision_tree.pkl"
    }
    
    model_file = model_dir / model_files.get(model_name)
    
    if not model_file.exists():
        return None
    
    try:
        return joblib.load(model_file)
    except Exception as e:
        print(f"Error loading {model_name} for {symbol} {horizon}d: {e}")
        return None


def get_ml_prediction(symbol: str, horizon: int, models: List[str] = None) -> Dict[str, Any]:
    """
    Get ML predictions for a symbol across multiple models.
    
    Args:
        symbol: Stock ticker
        horizon: Prediction horizon (3, 5, or 10 days)
        models: List of model names to use (default: all models)
    
    Returns:
        Dictionary with predictions and performance context
    """
    if not _ML_AVAILABLE:
        error_msg = "ML modules not available"
        if '_ML_IMPORT_ERROR' in globals():
            error_msg += f": {_ML_IMPORT_ERROR}"
        return {
            "error": error_msg,
            "symbol": symbol,
            "horizon": horizon
        }
    
    if models is None:
        models = ["Random Forest", "XGBoost", "Logistic Regression", "Decision Tree"]
    
    # Validate horizon
    if horizon not in [3, 5, 10]:
        return {
            "error": f"Invalid horizon: {horizon}. Must be 3, 5, or 10.",
            "symbol": symbol
        }
    
    try:
        # Fetch recent data for symbol and SPY (for beta/alpha features)
        df = fetch_daily_bars(symbol, "2020-01-01", str(dt.date.today()))
        spy_df = fetch_daily_bars("SPY", "2020-01-01", str(dt.date.today()))
        
        # Check if fetch_daily_bars returned an error or invalid data
        if df is None:
            return {
                "error": "Failed to fetch data",
                "symbol": symbol,
                "horizon": horizon
            }
        
        if isinstance(df, str):
            return {
                "error": f"Data fetch error: {df}",
                "symbol": symbol,
                "horizon": horizon
            }
        
        if not hasattr(df, 'shape') or len(df) < 100:
            return {
                "error": "Insufficient data",
                "symbol": symbol,
                "horizon": horizon
            }
        
        # Engineer features with timeout (GDELT can hang)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Feature engineering timed out")
        
        try:
            # Set 30-second timeout for feature engineering
            if hasattr(signal, 'SIGALRM'):  # Unix-like systems
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
            
            df_features = engineer_features(symbol, df, spy_df)
            
            # Cancel alarm
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                
        except TimeoutError:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return {
                "error": "Feature engineering timed out (GDELT fetch likely hanging). ML predictions unavailable.",
                "symbol": symbol,
                "horizon": horizon
            }
        except Exception as e:
            return {
                "error": f"Feature engineering error: {str(e)}",
                "symbol": symbol,
                "horizon": horizon
            }
        
        if df_features is None:
            return {
                "error": "Feature engineering returned None",
                "symbol": symbol,
                "horizon": horizon
            }
        
        if isinstance(df_features, str):
            return {
                "error": f"Feature engineering returned error: {df_features}",
                "symbol": symbol,
                "horizon": horizon
            }
        
        if not hasattr(df_features, 'shape') or len(df_features) == 0:
            return {
                "error": "Feature engineering returned empty data",
                "symbol": symbol,
                "horizon": horizon
            }
        
        # Get latest data point
        latest = df_features.iloc[-1]
        
        # Prepare feature vector (drop target and metadata columns)
        drop_cols = ['target', 'forward_return', 'date', 'symbol'] + \
                   [col for col in df_features.columns if col.startswith('target_') or col.startswith('forward_')]
        feature_cols = [col for col in df_features.columns if col not in drop_cols]
        
        X = latest[feature_cols].values.reshape(1, -1)
        
        # Get predictions from each model
        predictions = {}
        models_loaded = 0
        
        for model_name in models:
            model_data = load_model(symbol, horizon, model_name)
            
            if model_data is None:
                predictions[model_name] = {
                    "prediction": None,
                    "error": f"Model not found: ml_models/saved_models/{symbol}_{horizon}d/{model_name}"
                }
                continue
            
            models_loaded += 1
            
            try:
                # Extract the actual model and scaler from the loaded dict
                model = model_data['model']
                scaler = model_data.get('scaler')
                stored_features = model_data.get('feature_cols', feature_cols)
                
                # Check for missing features and handle gracefully
                # latest is a Series, so use .index instead of .columns
                available_features = set(latest.index)
                required_features = set(stored_features)
                missing_features = required_features - available_features
                
                if missing_features:
                    # Try to fill missing features with zeros (typical for missing sentiment)
                    for feat in missing_features:
                        latest[feat] = 0.0
                    
                    # Log warning for debugging
                    import warnings
                    warnings.warn(f"{model_name}: Filled {len(missing_features)} missing features with zeros (likely sentiment features)")
                
                # Prepare features in correct order
                X_prepared = latest[stored_features].values.reshape(1, -1)
                
                # Scale if needed
                if scaler is not None:
                    X_prepared = scaler.transform(X_prepared)
                
                # Get prediction (0 = down/flat, 1 = up)
                pred = model.predict(X_prepared)[0]
                
                # Get probability if available
                prob = None
                if hasattr(model, 'predict_proba'):
                    prob_array = model.predict_proba(X_prepared)[0]
                    prob = {
                        "down": float(prob_array[0]),
                        "up": float(prob_array[1]),
                        "confidence": float(max(prob_array))
                    }
                
                predictions[model_name] = {
                    "prediction": "UP" if pred == 1 else "DOWN/FLAT",
                    "direction": int(pred),
                    "probability": prob,
                    "performance": MODEL_PERFORMANCE[f"{horizon}d"]["models"][model_name],
                    "missing_features_filled": len(missing_features) if missing_features else 0
                }
            except Exception as e:
                predictions[model_name] = {
                    "prediction": None,
                    "error": str(e)
                }
        
        # Check if any models loaded successfully
        if models_loaded == 0:
            model_dir = ml_models_path / "saved_models" / f"{symbol}_{horizon}d"
            return {
                "error": f"No trained models found for {symbol} {horizon}d horizon. Expected directory: {model_dir}. You need to train models first by running: cd ml_models && python run_pipeline.py --symbols {symbol}",
                "symbol": symbol,
                "horizon": horizon
            }
        
        # Calculate consensus
        up_votes = sum(1 for p in predictions.values() 
                      if p.get("direction") == 1)
        total_votes = len([p for p in predictions.values() 
                          if p.get("direction") is not None])
        
        consensus_strength = up_votes / total_votes if total_votes > 0 else 0
        
        # Determine consensus direction
        if consensus_strength > 0.75:
            consensus = "STRONG UP"
        elif consensus_strength >= 0.5:
            consensus = "WEAK UP"
        elif consensus_strength >= 0.25:
            consensus = "WEAK DOWN"
        else:
            consensus = "STRONG DOWN"
        
        # Calculate agreement strength (toward consensus direction, 0.5 to 1.0)
        # This represents how unified the models are
        if consensus_strength >= 0.5:
            # UP consensus: agreement = how much above 50%
            agreement_strength = consensus_strength
        else:
            # DOWN consensus: agreement = how much below 50%
            agreement_strength = 1.0 - consensus_strength
        
        # Identify best model for this horizon
        best_model_name = MODEL_PERFORMANCE[f"{horizon}d"]["best_model"]
        best_prediction = predictions.get(best_model_name, {})
        
        return {
            "symbol": symbol,
            "horizon": f"{horizon}d",
            "timestamp": str(dt.datetime.now()),
            "predictions": predictions,
            "consensus": {
                "direction": consensus,
                "up_votes": up_votes,
                "total_votes": total_votes,
                "confidence": agreement_strength
            },
            "best_model": {
                "name": best_model_name,
                "prediction": best_prediction.get("prediction"),
                "probability": best_prediction.get("probability"),
                "performance": MODEL_PERFORMANCE[f"{horizon}d"]["models"][best_model_name]
            },
            "horizon_performance": MODEL_PERFORMANCE[f"{horizon}d"],
            "pipeline_context": PIPELINE_STATS,
            "latest_price": float(latest.get('close', 0)),
            "feature_count": len(feature_cols)
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n[ERROR] Exception in get_ml_prediction:")
        print(error_details)
        return {
            "error": str(e),
            "symbol": symbol,
            "horizon": horizon
        }


# Tool function for agent integration
def ml_prediction_tool_fn(state: dict, args: dict) -> dict:
    """
    Get ML model predictions with comprehensive performance context.
    
    Args (in args dict):
        symbol: Stock ticker (required)
        horizon: Prediction horizon in days - 3, 5, or 10 (default: 5)
        models: List of model names (default: all 4 models)
    
    Returns updated state with predictions in state["tool_results"]["ml_prediction"]
    """
    if "tool_results" not in state:
        state["tool_results"] = {}
    
    symbol = args.get("symbol")
    if not symbol:
        state["tool_results"]["ml_prediction"] = {
            "error": "Symbol is required"
        }
        return state
    
    horizon = args.get("horizon", 5)
    models = args.get("models")  # None = use all models
    
    result = get_ml_prediction(symbol, horizon, models)
    state["tool_results"]["ml_prediction"] = result
    
    return state


# For testing
if __name__ == "__main__":
    import datetime as dt
    
    print("Testing ML Prediction Tool")
    print("="*80)
    
    # Test AAPL 5-day prediction
    result = get_ml_prediction("AAPL", 5)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nSymbol: {result['symbol']}")
        print(f"Horizon: {result['horizon']}")
        print(f"\nConsensus: {result['consensus']['direction']}")
        print(f"  Confidence: {result['consensus']['confidence']:.1%}")
        print(f"  Up votes: {result['consensus']['up_votes']}/{result['consensus']['total_votes']}")
        
        print(f"\nBest Model ({result['best_model']['name']}):")
        print(f"  Prediction: {result['best_model']['prediction']}")
        print(f"  Sharpe: {result['best_model']['performance']['sharpe']:.2f}")
        
        print(f"\nAll Model Predictions:")
        for model_name, pred in result['predictions'].items():
            if 'prediction' in pred and pred['prediction']:
                prob_str = ""
                if pred.get('probability'):
                    prob_str = f" ({pred['probability']['confidence']:.1%} conf)"
                print(f"  {model_name:20s}: {pred['prediction']}{prob_str}")
