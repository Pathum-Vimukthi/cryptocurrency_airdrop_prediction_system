#!/usr/bin/env python3
"""
Combined Prediction Script - Using Trained XGBoost Models
Predicts scam detection, token rewards, and token price using actual trained models
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import traceback
import os
import warnings
warnings.filterwarnings('ignore')

# Print to stderr for debugging
def log_error(message):
    print(f"ERROR: {message}", file=sys.stderr)
    
def log_info(message):
    print(f"INFO: {message}", file=sys.stderr)

def load_models():
    """Load all trained XGBoost models"""
    try:
        log_info("Starting model loading...")
        
        # Check if files exist
        required_files = [
            'xgboost_scam_detector.json',
            'xgboost_scam_detector.pkl',
            'xgboost_reward_predictor.json',
            'xgboost_reward_predictor.pkl',
            'xgboost_price_predictor.json',
            'xgboost_price_predictor.pkl'
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file not found: {file}")
            log_info(f"Found: {file}")
        
        # Load Scam Detection Model
        log_info("Loading scam detection model...")
        scam_model = xgb.XGBClassifier()
        scam_model.load_model('xgboost_scam_detector.json')
        
        with open('xgboost_scam_detector.pkl', 'rb') as f:
            scam_package = pickle.load(f)
            scam_scaler = scam_package['scaler']
            scam_features = scam_package['feature_cols']
        log_info(f"Scam model loaded. Features: {len(scam_features)}")
        
        # Load Reward Prediction Model
        log_info("Loading reward prediction model...")
        reward_model = xgb.XGBRegressor()
        reward_model.load_model('xgboost_reward_predictor.json')
        
        with open('xgboost_reward_predictor.pkl', 'rb') as f:
            reward_package = pickle.load(f)
            reward_scaler = reward_package['scaler']
            reward_features = reward_package['feature_cols']
        log_info(f"Reward model loaded. Features: {len(reward_features)}")
        
        # Load Price Prediction Model
        log_info("Loading price prediction model...")
        price_model = xgb.XGBRegressor()
        price_model.load_model('xgboost_price_predictor.json')
        
        with open('xgboost_price_predictor.pkl', 'rb') as f:
            price_package = pickle.load(f)
            price_scaler = price_package['scaler']
            price_features = price_package['feature_cols']
        log_info(f"Price model loaded. Features: {len(price_features)}")
        
        log_info("All models loaded successfully!")
        
        return {
            'scam': {'model': scam_model, 'scaler': scam_scaler, 'features': scam_features},
            'reward': {'model': reward_model, 'scaler': reward_scaler, 'features': reward_features},
            'price': {'model': price_model, 'scaler': price_scaler, 'features': price_features}
        }
    except Exception as e:
        log_error(f"Model loading failed: {str(e)}")
        log_error(traceback.format_exc())
        raise Exception(f"Failed to load models: {str(e)}")

def engineer_features_scam(data):
    """Engineer features for scam detection model"""
    features = {}
    
    # Log transformations
    features['ICO_token_price_log'] = np.log10(float(data['ICO_token_price']) + 1e-10)
    features['Total_supply_log'] = np.log10(float(data['Total_supply']) + 1)
    
    # Calculate derived features
    features['Days_Per_Task'] = float(data['Days_Since_Published']) / (int(data['Task_Count']) + 1)
    features['Has_Marketing'] = 1 if (int(data['Has_Social']) == 1 or int(data['Has_Referral']) == 1) else 0
    features['Token_Value_Risk'] = features['ICO_token_price_log'] * features['Total_supply_log']
    features['Task_Social_Engagement'] = int(data['Task_Count']) * (int(data['Has_Social']) + 1)
    
    # Original features
    features['Published_Year'] = int(data['Published_Year'])
    features['Published_Month'] = int(data['Published_Month'])
    features['Days_Since_Published'] = float(data['Days_Since_Published'])
    features['Task_Count'] = int(data['Task_Count'])
    features['Has_Social'] = int(data['Has_Social'])
    features['Has_Referral'] = int(data['Has_Referral'])
    features['Tokens_per_referral'] = float(data['Tokens_per_referral'])
    features['Task_Complexity_Score'] = int(data['Task_Count']) * (int(data['Has_Social']) + int(data['Has_Referral']))
    
    return features

def engineer_features_reward(data):
    """Engineer features for reward prediction model"""
    features = {}
    
    # Log transformations
    features['Total_supply_log'] = np.log10(float(data['Total_supply']) + 1)
    features['ICO_token_price_log'] = np.log10(float(data['ICO_token_price']) + 1e-10)
    
    # Derived features
    features['Days_Per_Task'] = float(data['Days_Since_Published']) / (int(data['Task_Count']) + 1)
    features['Has_Marketing'] = 1 if (int(data['Has_Social']) == 1 or int(data['Has_Referral']) == 1) else 0
    features['Referral_Strength'] = float(data['Tokens_per_referral']) * int(data['Has_Referral'])
    features['Task_Engagement'] = int(data['Task_Count']) * (int(data['Has_Social']) + 1)
    features['Supply_Price_Ratio'] = features['Total_supply_log'] / (features['ICO_token_price_log'] + 1)
    features['Project_Maturity'] = np.log10(float(data['Days_Since_Published']) + 1)
    
    # Original features
    features['Tokens_per_referral'] = float(data['Tokens_per_referral'])
    features['Task_Count'] = int(data['Task_Count'])
    features['Has_Referral'] = int(data['Has_Referral'])
    features['Has_Social'] = int(data['Has_Social'])
    features['Published_Year'] = int(data['Published_Year'])
    features['Days_Since_Published'] = float(data['Days_Since_Published'])
    
    return features

def engineer_features_price(data, predicted_tokens):
    """Engineer features for price prediction model"""
    features = {}
    
    # Log transformations
    features['Total_supply_log'] = np.log10(float(data['Total_supply']) + 1)
    features['Tokens_per_airdrop_log'] = np.log10(predicted_tokens + 1)
    
    # Derived features
    features['Days_Per_Task'] = float(data['Days_Since_Published']) / (int(data['Task_Count']) + 1)
    features['Has_Marketing'] = 1 if (int(data['Has_Social']) == 1 or int(data['Has_Referral']) == 1) else 0
    features['Referral_Strength'] = float(data['Tokens_per_referral']) * int(data['Has_Referral'])
    features['Task_Engagement'] = int(data['Task_Count']) * (int(data['Has_Social']) + 1)
    features['Supply_Reward_Ratio'] = features['Total_supply_log'] / (features['Tokens_per_airdrop_log'] + 1)
    features['Project_Maturity'] = np.log10(float(data['Days_Since_Published']) + 1)
    
    # Original features
    features['Tokens_per_referral'] = float(data['Tokens_per_referral'])
    features['Task_Count'] = int(data['Task_Count'])
    features['Has_Referral'] = int(data['Has_Referral'])
    features['Has_Social'] = int(data['Has_Social'])
    features['Published_Year'] = int(data['Published_Year'])
    features['Days_Since_Published'] = float(data['Days_Since_Published'])
    
    return features

def predict_scam(input_data, models):
    """Predict scam using trained XGBoost model"""
    try:
        # Engineer features
        features = engineer_features_scam(input_data)
        
        # Create DataFrame with correct feature order
        feature_order = models['scam']['features']
        X = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        
        # Scale features
        X_scaled = models['scam']['scaler'].transform(X)
        
        # Make prediction
        prediction = models['scam']['model'].predict(X_scaled)[0]
        probabilities = models['scam']['model'].predict_proba(X_scaled)[0]
        
        is_scam = bool(prediction)
        scam_probability = float(probabilities[1])
        legitimate_probability = float(probabilities[0])
        
        # Calculate risk score (0-100)
        risk_score = int(scam_probability * 100)
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "Critical"
            color = "#d32f2f"
        elif risk_score >= 50:
            risk_level = "High"
            color = "#f57c00"
        elif risk_score >= 30:
            risk_level = "Medium"
            color = "#fbc02d"
        else:
            risk_level = "Low"
            color = "#388e3c"
        
        # Calculate warning flags
        reward_ratio = float(input_data['Tokens_per_referral']) / 100
        supply_inflation = float(input_data['Total_supply']) / (float(input_data['ICO_token_price']) * 1_000_000 + 1)
        project_age = float(input_data['Days_Since_Published']) / 365
        
        return {
            'is_scam': is_scam,
            'scam_probability': round(scam_probability, 4),
            'legitimate_probability': round(legitimate_probability, 4),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': color,
            'confidence': round(max(scam_probability, legitimate_probability), 4),
            'warning_flags': {
                'high_reward_ratio': reward_ratio > 1.5,
                'high_inflation_risk': supply_inflation > 5000,
                'very_new_project': project_age < 0.5,
                'no_social_media': int(input_data['Has_Social']) == 0,
                'suspicious_tasks': int(input_data['Task_Count']) == 0 and int(input_data['Has_Referral']) == 1
            }
        }
    except Exception as e:
        raise Exception(f"Scam prediction failed: {str(e)}")

def predict_reward(input_data, models):
    """Predict reward using trained XGBoost model"""
    try:
        # Engineer features
        features = engineer_features_reward(input_data)
        
        # Create DataFrame with correct feature order
        feature_order = models['reward']['features']
        X = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        
        # Scale features
        X_scaled = models['reward']['scaler'].transform(X)
        
        # Make prediction (log scale)
        prediction_log = models['reward']['model'].predict(X_scaled)[0]
        
        # Convert back to original scale
        predicted_tokens = max(0, 10**prediction_log - 1)
        
        # Calculate additional metrics
        referral_bonus = float(input_data['Tokens_per_referral']) if int(input_data['Has_Referral']) == 1 else 0
        total_earning_potential = predicted_tokens + referral_bonus
        task_count = int(input_data['Task_Count'])
        earning_efficiency = total_earning_potential / (task_count + 1)
        tokens_per_task = predicted_tokens / (task_count + 1)
        
        return {
            'predicted_tokens': round(predicted_tokens, 2),
            'referral_bonus': round(referral_bonus, 2),
            'total_earning_potential': round(total_earning_potential, 2),
            'earning_efficiency': round(earning_efficiency, 2),
            'tasks_required': task_count,
            'tokens_per_task': round(tokens_per_task, 2)
        }
    except Exception as e:
        raise Exception(f"Reward prediction failed: {str(e)}")

def predict_price(input_data, predicted_tokens, models):
    """Predict price using trained XGBoost model"""
    try:
        # Engineer features
        features = engineer_features_price(input_data, predicted_tokens)
        
        # Create DataFrame with correct feature order
        feature_order = models['price']['features']
        X = pd.DataFrame([[features[f] for f in feature_order]], columns=feature_order)
        
        # Scale features
        X_scaled = models['price']['scaler'].transform(X)
        
        # Make prediction (log scale)
        prediction_log = models['price']['model'].predict(X_scaled)[0]
        
        # Convert back to original scale
        predicted_price = max(0, 10**prediction_log - 1e-10)
        
        ico_price = float(input_data['ICO_token_price'])
        total_supply = float(input_data['Total_supply'])
        
        # Calculate metrics
        price_change_percent = ((predicted_price - ico_price) / ico_price) * 100
        market_cap = total_supply * predicted_price
        
        # Calculate stability score
        project_age = float(input_data['Days_Since_Published']) / 365
        if project_age > 2 and int(input_data['Has_Social']) == 1:
            stability_score = 0.85
        elif project_age > 1:
            stability_score = 0.70
        else:
            stability_score = 0.50
        
        return {
            'predicted_price_usd': round(predicted_price, 6),
            'ico_price_usd': round(ico_price, 6),
            'price_change_percent': round(price_change_percent, 2),
            'market_cap_estimate': round(market_cap, 2),
            'stability_score': round(stability_score, 2),
            'price_trend': 'Bullish' if predicted_price > ico_price else 'Bearish'
        }
    except Exception as e:
        raise Exception(f"Price prediction failed: {str(e)}")

def main():
    try:
        log_info("="*60)
        log_info("Starting prediction process...")
        
        # Check if argument provided
        if len(sys.argv) < 2:
            raise ValueError("No input data provided. Usage: python3 predict_all.py '<json_data>'")
        
        # Parse input JSON
        input_json = sys.argv[1]
        log_info(f"Received input: {input_json[:100]}...")
        
        input_data = json.loads(input_json)
        log_info(f"Parsed input data: {list(input_data.keys())}")
        
        # Validate required fields
        required_fields = [
            'ICO_token_price', 'Total_supply', 'Published_Year', 'Published_Month',
            'Days_Since_Published', 'Task_Count', 'Has_Social', 'Has_Referral',
            'Tokens_per_referral'
        ]
        
        missing_fields = [f for f in required_fields if f not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        log_info("All required fields present")
        
        # Load all models
        log_info("Loading models...")
        models = load_models()
        log_info("Models loaded successfully")
        
        # Run predictions
        log_info("Running scam detection...")
        scam_result = predict_scam(input_data, models)
        log_info(f"Scam detection complete. Risk: {scam_result['risk_level']}")
        
        log_info("Running reward prediction...")
        reward_result = predict_reward(input_data, models)
        log_info(f"Reward prediction complete. Tokens: {reward_result['predicted_tokens']}")
        
        log_info("Running price prediction...")
        price_result = predict_price(input_data, reward_result['predicted_tokens'], models)
        log_info(f"Price prediction complete. Price: ${price_result['predicted_price_usd']}")
        
        # Calculate expected value
        expected_value_usd = reward_result['total_earning_potential'] * price_result['predicted_price_usd']
        log_info(f"Expected value: ${expected_value_usd:.2f}")
        
        # Generate recommendation
        risk_score = scam_result['risk_score']
        
        if risk_score >= 50:
            action = "AVOID"
            priority = "Do Not Participate"
            roi_estimate = "High Risk"
            reasoning = f"High scam risk ({risk_score}%). Model confidence: {scam_result['confidence']*100:.1f}%. Not recommended."
        elif risk_score >= 30:
            action = "CAUTION"
            priority = "Low Priority"
            roi_estimate = "Moderate Risk"
            reasoning = f"Medium risk ({risk_score}%). Model confidence: {scam_result['confidence']*100:.1f}%. Proceed with caution."
        else:
            if expected_value_usd > 100:
                action = "PARTICIPATE"
                priority = "High Priority"
                roi_estimate = "Excellent"
            elif expected_value_usd > 10:
                action = "PARTICIPATE"
                priority = "Medium Priority"
                roi_estimate = "Good"
            else:
                action = "CONSIDER"
                priority = "Low Priority"
                roi_estimate = "Fair"
            
            reasoning = f"Low scam risk ({risk_score}%). Expected value: ${expected_value_usd:.2f}. Model confidence: {scam_result['confidence']*100:.1f}%."
        
        # Combine results
        result = {
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_summary': {
                'project_age_years': round(float(input_data['Days_Since_Published']) / 365, 2),
                'has_social_media': bool(int(input_data['Has_Social'])),
                'has_referral_program': bool(int(input_data['Has_Referral'])),
                'total_tasks': int(input_data['Task_Count'])
            },
            'scam_detection': scam_result,
            'reward_prediction': reward_result,
            'price_prediction': price_result,
            'investment_summary': {
                'expected_value_usd': round(expected_value_usd, 2),
                'roi_estimate': roi_estimate,
                'time_investment': f"{input_data['Task_Count']} tasks",
                'earning_per_task': round(expected_value_usd / (int(input_data['Task_Count']) + 1), 2)
            },
            'recommendation': {
                'action': action,
                'priority': priority,
                'confidence': scam_result['confidence'],
                'roi_estimate': roi_estimate,
                'reasoning': reasoning
            }
        }
        
        log_info("Prediction complete! Outputting results...")
        log_info("="*60)
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        log_error("="*60)
        log_error(f"FATAL ERROR: {str(e)}")
        log_error(traceback.format_exc())
        log_error("="*60)
        
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()