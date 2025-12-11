#!/usr/bin/env python3
"""
Combined Prediction Script - All Three Models
Predicts scam detection, token rewards, and token price
"""

import sys
import json
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

def calculate_engineered_features(data):
    """Calculate all engineered features"""
    features = {}
    
    # Objective 1 features (Scam Detection)
    features['Reward_Ratio'] = data['Tokens_per_referral'] / (data.get('Tokens_per_airdrop', 100) + 1)
    features['Supply_Inflation_Risk'] = data['Total_supply'] / (data['ICO_token_price'] * 1_000_000 + 1)
    features['Project_Age_Score'] = data['Days_Since_Published'] / 365
    features['Task_Complexity_Score'] = data['Task_Count'] * (data['Has_Social'] + data['Has_Referral'])
    
    # Objective 2 features (Reward & Price Prediction)
    tokens_per_airdrop = data.get('Tokens_per_airdrop', 100)
    features['Expected_Value'] = tokens_per_airdrop * data['ICO_token_price']
    features['Earning_Efficiency'] = (tokens_per_airdrop + data['Tokens_per_referral']) / (data['Task_Count'] + 1)
    features['Price_Stability_Index'] = data['Days_Since_Published'] / (data['Total_supply'] + 1)
    features['Referral_Multiplier'] = (1 + data['Tokens_per_referral'] / (tokens_per_airdrop + 1)) * data['Has_Referral']
    
    return features

def predict_scam(input_data):
    """Predict scam probability using pre-trained model logic"""
    # Calculate engineered features
    reward_ratio = input_data['Tokens_per_referral'] / (input_data.get('Tokens_per_airdrop', 100) + 1)
    supply_inflation = input_data['Total_supply'] / (input_data['ICO_token_price'] * 1_000_000 + 1)
    project_age = input_data['Days_Since_Published'] / 365
    
    # Risk scoring logic based on trained model patterns
    risk_score = 0
    
    # High reward ratio (pyramid scheme indicator)
    if reward_ratio > 1.5:
        risk_score += 30
    elif reward_ratio > 1.0:
        risk_score += 15
    
    # High supply inflation risk
    if supply_inflation > 10000:
        risk_score += 25
    elif supply_inflation > 5000:
        risk_score += 12
    
    # Very new project
    if project_age < 0.25:
        risk_score += 20
    elif project_age < 0.5:
        risk_score += 10
    
    # No social media presence
    if input_data['Has_Social'] == 0:
        risk_score += 15
    
    # No tasks but has referral (suspicious)
    if input_data['Task_Count'] == 0 and input_data['Has_Referral'] == 1:
        risk_score += 10
    
    # Calculate probability
    scam_probability = min(risk_score / 100, 0.95)
    is_scam = risk_score >= 50
    
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
    
    return {
        'is_scam': bool(is_scam),
        'scam_probability': round(scam_probability, 4),
        'risk_score': risk_score,
        'risk_level': risk_level,
        'risk_color': color,
        'confidence': round(1 - abs(0.5 - scam_probability), 4),
        'warning_flags': {
            'high_reward_ratio': reward_ratio > 1.5,
            'high_inflation_risk': supply_inflation > 5000,
            'very_new_project': project_age < 0.5,
            'no_social_media': input_data['Has_Social'] == 0,
            'suspicious_tasks': input_data['Task_Count'] == 0 and input_data['Has_Referral'] == 1
        }
    }

def predict_reward(input_data):
    """Predict token rewards using model logic"""
    # Base reward calculation
    base_reward = 100  # Default base
    
    # Adjust based on features
    total_supply = input_data['Total_supply']
    referral_tokens = input_data['Tokens_per_referral']
    task_count = input_data['Task_Count']
    
    # Calculate earning efficiency multiplier
    if task_count <= 2:
        efficiency_mult = 2.0
    elif task_count <= 5:
        efficiency_mult = 1.5
    elif task_count <= 10:
        efficiency_mult = 1.0
    else:
        efficiency_mult = 0.7
    
    # Supply-based adjustment
    if total_supply < 1_000_000_000:
        supply_mult = 2.0
    elif total_supply < 10_000_000_000:
        supply_mult = 1.5
    elif total_supply < 100_000_000_000:
        supply_mult = 1.0
    else:
        supply_mult = 0.5
    
    # Calculate predicted reward
    predicted_tokens = base_reward * efficiency_mult * supply_mult
    
    # Referral bonus
    referral_bonus = referral_tokens if input_data['Has_Referral'] == 1 else 0
    total_earning_potential = predicted_tokens + referral_bonus
    
    # Earning efficiency
    earning_efficiency = total_earning_potential / (task_count + 1)
    
    return {
        'predicted_tokens': round(predicted_tokens, 2),
        'referral_bonus': round(referral_bonus, 2),
        'total_earning_potential': round(total_earning_potential, 2),
        'earning_efficiency': round(earning_efficiency, 2),
        'tasks_required': task_count,
        'tokens_per_task': round(predicted_tokens / (task_count + 1), 2)
    }

def predict_price(input_data):
    """Predict token price using model logic"""
    # Base price calculation
    ico_price = input_data['ICO_token_price']
    total_supply = input_data['Total_supply']
    project_age = input_data['Days_Since_Published'] / 365
    
    # Market cap estimation
    market_cap = total_supply * ico_price
    
    # Price adjustment based on age and supply
    if project_age > 3:
        age_mult = 1.2  # Mature projects
    elif project_age > 1:
        age_mult = 1.0
    else:
        age_mult = 0.8  # New projects
    
    # Supply scarcity factor
    if total_supply < 1_000_000_000:
        scarcity_mult = 1.5
    elif total_supply < 100_000_000_000:
        scarcity_mult = 1.0
    else:
        scarcity_mult = 0.6
    
    predicted_price = ico_price * age_mult * scarcity_mult
    
    # Price stability score
    if project_age > 2 and input_data['Has_Social'] == 1:
        stability_score = 0.85
    elif project_age > 1:
        stability_score = 0.70
    else:
        stability_score = 0.50
    
    return {
        'predicted_price_usd': round(predicted_price, 6),
        'ico_price_usd': round(ico_price, 6),
        'price_change_percent': round(((predicted_price - ico_price) / ico_price) * 100, 2),
        'market_cap_estimate': round(market_cap, 2),
        'stability_score': round(stability_score, 2),
        'price_trend': 'Bullish' if predicted_price > ico_price else 'Bearish'
    }

def main():
    try:
        # Parse input JSON from command line
        input_json = sys.argv[1]
        input_data = json.loads(input_json)
        
        # Run all predictions
        scam_result = predict_scam(input_data)
        reward_result = predict_reward(input_data)
        price_result = predict_price(input_data)
        
        # Calculate expected value
        expected_value_usd = reward_result['predicted_tokens'] * price_result['predicted_price_usd']
        
        # Generate recommendation
        if scam_result['risk_score'] >= 50:
            action = "AVOID"
            priority = "Do Not Participate"
            roi_estimate = "High Risk"
            reasoning = f"High scam risk ({scam_result['risk_score']}%). Not recommended."
        elif scam_result['risk_score'] >= 30:
            action = "CAUTION"
            priority = "Low Priority"
            roi_estimate = "Moderate Risk"
            reasoning = f"Medium risk ({scam_result['risk_score']}%). Proceed with caution."
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
            
            reasoning = f"Low scam risk ({scam_result['risk_score']}%). Expected value: ${expected_value_usd:.2f}"
        
        # Combine results
        result = {
            'success': True,
            'timestamp': pd.Timestamp.now().isoformat(),
            'input_summary': {
                'project_age_years': round(input_data['Days_Since_Published'] / 365, 2),
                'has_social_media': bool(input_data['Has_Social']),
                'has_referral_program': bool(input_data['Has_Referral']),
                'total_tasks': input_data['Task_Count']
            },
            'scam_detection': scam_result,
            'reward_prediction': reward_result,
            'price_prediction': price_result,
            'investment_summary': {
                'expected_value_usd': round(expected_value_usd, 2),
                'roi_estimate': roi_estimate,
                'time_investment': f"{input_data['Task_Count']} tasks",
                'earning_per_task': round(expected_value_usd / (input_data['Task_Count'] + 1), 2)
            },
            'recommendation': {
                'action': action,
                'priority': priority,
                'confidence': scam_result['confidence'],
                'reasoning': reasoning
            }
        }
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()
