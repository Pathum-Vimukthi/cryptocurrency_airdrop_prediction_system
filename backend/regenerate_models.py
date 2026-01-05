"""
Quick script to regenerate model pickle files with current NumPy version
Run this in the directory where your training script generated the models
"""

import pickle
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

print("="*80)
print("REGENERATING MODEL PICKLE FILES")
print("="*80)

print(f"\nCurrent NumPy version: {np.__version__}")

# ============================================================================
# SCAM DETECTION MODEL
# ============================================================================
print("\n[1] Regenerating Scam Detection Model...")

try:
    # Load XGBoost model (JSON format - no NumPy dependency)
    scam_model = xgb.XGBClassifier()
    scam_model.load_model('xgboost_scam_detector.json')
    print("    ✓ Loaded xgboost_scam_detector.json")
    
    # Try to load old pickle to get metadata
    try:
        with open('xgboost_scam_detector.pkl', 'rb') as f:
            old_package = pickle.load(f)
            scaler = old_package['scaler']
            feature_cols = old_package['feature_cols']
            metrics = old_package.get('metrics', {})
            model_params = old_package.get('model_params', {})
        print("    ✓ Loaded metadata from old pickle file")
    except:
        print("    ⚠ Could not load old pickle, creating new scaler...")
        # Create new scaler if can't load old one
        scaler = StandardScaler()
        feature_cols = [
            'ICO_token_price_log',
            'Total_supply_log',
            'Published_Year',
            'Published_Month',
            'Days_Since_Published',
            'Task_Count',
            'Has_Social',
            'Has_Referral',
            'Tokens_per_referral',
            'Task_Complexity_Score',
            'Days_Per_Task',
            'Has_Marketing',
            'Token_Value_Risk',
            'Task_Social_Engagement'
        ]
        metrics = {}
        model_params = {}
    
    # Create new package with current NumPy
    new_package = {
        'model': scam_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'model_params': model_params
    }
    
    # Save with current NumPy version
    with open('xgboost_scam_detector.pkl', 'wb') as f:
        pickle.dump(new_package, f)
    
    print("    ✓ Regenerated xgboost_scam_detector.pkl")
    
except Exception as e:
    print(f"    ✗ Failed: {e}")

# ============================================================================
# REWARD PREDICTION MODEL
# ============================================================================
print("\n[2] Regenerating Reward Prediction Model...")

try:
    # Load XGBoost model
    reward_model = xgb.XGBRegressor()
    reward_model.load_model('xgboost_reward_predictor.json')
    print("    ✓ Loaded xgboost_reward_predictor.json")
    
    # Try to load old pickle
    try:
        with open('xgboost_reward_predictor.pkl', 'rb') as f:
            old_package = pickle.load(f)
            scaler = old_package['scaler']
            feature_cols = old_package['feature_cols']
            metrics = old_package.get('metrics', {})
            model_params = old_package.get('model_params', {})
        print("    ✓ Loaded metadata from old pickle file")
    except:
        print("    ⚠ Could not load old pickle, creating new scaler...")
        scaler = RobustScaler()
        feature_cols = [
            'Total_supply_log',
            'ICO_token_price_log',
            'Tokens_per_referral',
            'Task_Count',
            'Has_Referral',
            'Has_Social',
            'Published_Year',
            'Days_Since_Published',
            'Days_Per_Task',
            'Has_Marketing',
            'Referral_Strength',
            'Task_Engagement',
            'Supply_Price_Ratio',
            'Project_Maturity'
        ]
        metrics = {}
        model_params = {}
    
    # Create new package
    new_package = {
        'model': reward_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'model_params': model_params
    }
    
    # Save
    with open('xgboost_reward_predictor.pkl', 'wb') as f:
        pickle.dump(new_package, f)
    
    print("    ✓ Regenerated xgboost_reward_predictor.pkl")
    
except Exception as e:
    print(f"    ✗ Failed: {e}")

# ============================================================================
# PRICE PREDICTION MODEL
# ============================================================================
print("\n[3] Regenerating Price Prediction Model...")

try:
    # Load XGBoost model
    price_model = xgb.XGBRegressor()
    price_model.load_model('xgboost_price_predictor.json')
    print("    ✓ Loaded xgboost_price_predictor.json")
    
    # Try to load old pickle
    try:
        with open('xgboost_price_predictor.pkl', 'rb') as f:
            old_package = pickle.load(f)
            scaler = old_package['scaler']
            feature_cols = old_package['feature_cols']
            metrics = old_package.get('metrics', {})
            model_params = old_package.get('model_params', {})
        print("    ✓ Loaded metadata from old pickle file")
    except:
        print("    ⚠ Could not load old pickle, creating new scaler...")
        scaler = RobustScaler()
        feature_cols = [
            'Total_supply_log',
            'Tokens_per_airdrop_log',
            'Tokens_per_referral',
            'Task_Count',
            'Has_Referral',
            'Has_Social',
            'Published_Year',
            'Days_Since_Published',
            'Days_Per_Task',
            'Has_Marketing',
            'Referral_Strength',
            'Task_Engagement',
            'Supply_Reward_Ratio',
            'Project_Maturity'
        ]
        metrics = {}
        model_params = {}
    
    # Create new package
    new_package = {
        'model': price_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'model_params': model_params
    }
    
    # Save
    with open('xgboost_price_predictor.pkl', 'wb') as f:
        pickle.dump(new_package, f)
    
    print("    ✓ Regenerated xgboost_price_predictor.pkl")
    
except Exception as e:
    print(f"    ✗ Failed: {e}")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print("\nTesting if files can be loaded...")

try:
    # Test scam model
    with open('xgboost_scam_detector.pkl', 'rb') as f:
        pkg = pickle.load(f)
    print("✓ xgboost_scam_detector.pkl - OK")
    
    # Test reward model
    with open('xgboost_reward_predictor.pkl', 'rb') as f:
        pkg = pickle.load(f)
    print("✓ xgboost_reward_predictor.pkl - OK")
    
    # Test price model
    with open('xgboost_price_predictor.pkl', 'rb') as f:
        pkg = pickle.load(f)
    print("✓ xgboost_price_predictor.pkl - OK")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS REGENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nYou can now run predict_all.py")
    
except Exception as e:
    print(f"\n✗ Verification failed: {e}")
    print("\nPlease re-run your training scripts to generate new models.")

print("\n")