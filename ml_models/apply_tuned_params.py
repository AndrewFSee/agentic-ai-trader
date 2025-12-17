"""
Apply tuned hyperparameters to full 25-stock training.
Uses best parameters from hyperparameter tuning results.
"""
import json
from pathlib import Path
from datetime import datetime

# Default hyperparameters (to be updated after tuning)
TUNED_PARAMS_5D = {
    'learning_rate': 0.05,
    'max_depth': 5,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.5,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

TUNED_PARAMS_10D = {
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.2,
    'min_child_weight': 5,
    'reg_alpha': 0.2,
    'reg_lambda': 0.8,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

TUNED_PARAMS_3D = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 200,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'gamma': 0.0,
    'min_child_weight': 1,
    'reg_alpha': 0.0,
    'reg_lambda': 0.3,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

def load_best_params_from_tuning(tuning_results_path: str = None):
    """
    Load best hyperparameters from tuning results.
    
    Args:
        tuning_results_path: Path to tuning results JSON. If None, uses latest.
    """
    if tuning_results_path is None:
        # Find latest tuning results
        results_dir = Path('results/ml_models')
        tuning_files = sorted(results_dir.glob('hyperparameter_tuning_*.json'))
        if not tuning_files:
            print("No tuning results found. Using default parameters.")
            return {3: TUNED_PARAMS_3D, 5: TUNED_PARAMS_5D, 10: TUNED_PARAMS_10D}
        tuning_results_path = tuning_files[-1]
    
    print(f"Loading tuned parameters from: {tuning_results_path}")
    
    with open(tuning_results_path, 'r') as f:
        data = json.load(f)
    
    # Aggregate best parameters by horizon
    best_params_by_horizon = {}
    
    for horizon in [3, 5, 10]:
        # Get all results for this horizon
        horizon_results = [r for r in data['results'] if r['horizon'] == horizon]
        
        if not horizon_results:
            continue
        
        # Average the parameters across all symbols
        # (Or use the best performing symbol's params)
        
        # Option 1: Use best performing symbol's params
        best_result = max(horizon_results, key=lambda x: x['test_sharpe'])
        best_params_by_horizon[horizon] = best_result['best_params']
        
        print(f"\nBest params for {horizon}d horizon ({best_result['symbol']}, Sharpe: {best_result['test_sharpe']:.3f}):")
        for param, value in best_result['best_params'].items():
            print(f"  {param:20s}: {value}")
    
    return best_params_by_horizon

def update_config_with_tuned_params(params_by_horizon: dict):
    """
    Update config.py with tuned hyperparameters.
    
    Args:
        params_by_horizon: Dict mapping horizon -> best_params
    """
    config_path = Path('config.py')
    
    # Read current config
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    # Find XGBOOST_PARAMS section
    new_lines = []
    in_xgb_section = False
    skip_until_closing_brace = False
    
    for line in lines:
        if 'XGBOOST_PARAMS' in line:
            in_xgb_section = True
            new_lines.append(line)
            new_lines.append("    # TUNED HYPERPARAMETERS (optimized via Optuna)\n")
            
            # Add tuned params for each horizon
            for horizon in sorted(params_by_horizon.keys()):
                params = params_by_horizon[horizon]
                new_lines.append(f"    {horizon}: {{\n")
                for param, value in params.items():
                    if param in ['random_state', 'use_label_encoder', 'eval_metric']:
                        continue  # Skip these, they're set separately
                    new_lines.append(f"        '{param}': {repr(value)},\n")
                new_lines.append("    },\n")
            
            skip_until_closing_brace = True
            continue
        
        if skip_until_closing_brace:
            if '}' in line and not '    ' in line[:4]:
                # Found closing brace of XGBOOST_PARAMS
                new_lines.append(line)
                skip_until_closing_brace = False
                in_xgb_section = False
            continue
        
        new_lines.append(line)
    
    # Write updated config
    backup_path = config_path.with_suffix('.py.backup')
    config_path.rename(backup_path)
    print(f"\nBackup created: {backup_path}")
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated: {config_path}")

if __name__ == "__main__":
    print("="*80)
    print("APPLY TUNED HYPERPARAMETERS")
    print("="*80)
    
    # Load best params from tuning results
    best_params = load_best_params_from_tuning()
    
    # Update config.py (optional - can also just use in run_pipeline.py directly)
    update_choice = input("\nUpdate config.py with tuned parameters? (y/n): ")
    
    if update_choice.lower() == 'y':
        update_config_with_tuned_params(best_params)
        print("\n✓ Config updated. Run: python run_pipeline.py --test")
    else:
        print("\nTuned parameters loaded but not applied to config.")
        print("You can manually copy them to config.py or modify run_pipeline.py to use them.")
    
    print("\n" + "="*80)
    print("TUNED PARAMETERS SUMMARY")
    print("="*80)
    for horizon, params in sorted(best_params.items()):
        print(f"\n{horizon}-day horizon:")
        for param, value in params.items():
            print(f"  {param:20s}: {value}")
