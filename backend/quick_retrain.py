"""
Quick Model Retraining Script
Trains all 3 models with sample data and saves them properly
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK MODEL RETRAINING")
print("="*80)

# ============================================================================
# CREATE SAMPLE TRAINING DATA
# ============================================================================
print("\n[1] Creating sample training data...")

np.random.seed(42)
n_samples = 1000

# Generate realistic synthetic data
data = {
    'ICO_token_price': np.random.uniform(0.0001, 10, n_samples),
    'Total_supply': np.random.choice([1e6, 1e7, 1e8, 1e9, 1e10], n_samples),
    'Published_Year': np.random.choice([2019, 2020, 2021, 2022, 2023, 2024], n_samples),
    'Published_Month': np.random.randint(1, 13, n_samples),
    'Days_Since_Published': np.random.randint(1, 2000, n_samples),
    'Task_Count': np.random.randint(0, 20, n_samples),
    'Has_Social': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'Has_Referral': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
    'Tokens_per_referral': np.random.uniform(1, 200, n_samples),
    'Task_Complexity_Score': np.random.randint(0, 30, n_samples)
}

df = pd.DataFrame(data)

# Create target for scam detection
scam_score = (
    (df['Tokens_per_referral'] / 50 * 0.3) +
    (df['Task_Count'] == 0).astype(float) * 0.2 +
    (df['Has_Social'] == 0).astype(float) * 0.2 +
    (df['Days_Since_Published'] < 180).astype(float) * 0.3
)
df['is_scam'] = (scam_score + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

# Create target for rewards (realistic token amounts)
df['Tokens_per_airdrop'] = np.clip(
    50 * (df['Task_Count'] + 1) * (df['Total_supply'] / 1e9) * np.random.uniform(0.5, 2, n_samples),
    10, 100000
)

print(f"    Generated {n_samples} samples")
print(f"    Scam ratio: {df['is_scam'].mean()*100:.1f}%")

# ============================================================================
# OBJECTIVE 1: SCAM DETECTION MODEL
# ============================================================================
print("\n" + "="*80)
print("OBJECTIVE 1: SCAM DETECTION MODEL")
print("="*80)

# Feature engineering for scam detection
df_scam = df.copy()
df_scam['ICO_token_price_log'] = np.log10(df_scam['ICO_token_price'] + 1e-10)
df_scam['Total_supply_log'] = np.log10(df_scam['Total_supply'] + 1)
df_scam['Days_Per_Task'] = df_scam['Days_Since_Published'] / (df_scam['Task_Count'] + 1)
df_scam['Has_Marketing'] = ((df_scam['Has_Social'] == 1) | (df_scam['Has_Referral'] == 1)).astype(int)
df_scam['Token_Value_Risk'] = df_scam['ICO_token_price_log'] * df_scam['Total_supply_log']
df_scam['Task_Social_Engagement'] = df_scam['Task_Count'] * (df_scam['Has_Social'] + 1)

feature_cols_scam = [
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

X_scam = df_scam[feature_cols_scam]
y_scam = df_scam['is_scam']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_scam, y_scam, test_size=0.25, random_state=42, stratify=y_scam
)

print(f"\n[2] Training Scam Detection Model...")
print(f"    Train: {len(X_train_s)}, Test: {len(X_test_s)}")

# Scale features
scaler_scam = StandardScaler()
X_train_s_scaled = scaler_scam.fit_transform(X_train_s)
X_test_s_scaled = scaler_scam.transform(X_test_s)

# Train model
xgb_scam = xgb.XGBClassifier(
    objective='binary:logistic',
    max_depth=4,
    learning_rate=0.03,
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

xgb_scam.fit(X_train_s_scaled, y_train_s, verbose=False)

# Calculate metrics
from sklearn.metrics import accuracy_score, f1_score
y_pred_s = xgb_scam.predict(X_test_s_scaled)
accuracy = accuracy_score(y_test_s, y_pred_s)
f1 = f1_score(y_test_s, y_pred_s)

print(f"    Accuracy: {accuracy*100:.2f}%")
print(f"    F1-Score: {f1:.4f}")

# Save model
xgb_scam.save_model('xgboost_scam_detector.json')
print(f"    Saved: xgboost_scam_detector.json")

scam_package = {
    'model': xgb_scam,
    'scaler': scaler_scam,
    'feature_cols': feature_cols_scam,
    'metrics': {'accuracy': accuracy, 'f1_score': f1}
}

with open('xgboost_scam_detector.pkl', 'wb') as f:
    pickle.dump(scam_package, f)
print(f"    Saved: xgboost_scam_detector.pkl")

# ============================================================================
# OBJECTIVE 2A: REWARD PREDICTION MODEL
# ============================================================================
print("\n" + "="*80)
print("OBJECTIVE 2A: REWARD PREDICTION MODEL")
print("="*80)

# Feature engineering for reward prediction
df_reward = df.copy()
df_reward['Total_supply_log'] = np.log10(df_reward['Total_supply'] + 1)
df_reward['ICO_token_price_log'] = np.log10(df_reward['ICO_token_price'] + 1e-10)
df_reward['Days_Per_Task'] = df_reward['Days_Since_Published'] / (df_reward['Task_Count'] + 1)
df_reward['Has_Marketing'] = ((df_reward['Has_Social'] == 1) | (df_reward['Has_Referral'] == 1)).astype(int)
df_reward['Referral_Strength'] = df_reward['Tokens_per_referral'] * df_reward['Has_Referral']
df_reward['Task_Engagement'] = df_reward['Task_Count'] * (df_reward['Has_Social'] + 1)
df_reward['Supply_Price_Ratio'] = df_reward['Total_supply_log'] / (df_reward['ICO_token_price_log'] + 1)
df_reward['Project_Maturity'] = np.log10(df_reward['Days_Since_Published'] + 1)

# Log transform target
df_reward['tokens_reward_log'] = np.log10(df_reward['Tokens_per_airdrop'] + 1)

feature_cols_reward = [
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

X_reward = df_reward[feature_cols_reward]
y_reward = df_reward['tokens_reward_log']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reward, y_reward, test_size=0.25, random_state=42
)

print(f"\n[3] Training Reward Prediction Model...")
print(f"    Train: {len(X_train_r)}, Test: {len(X_test_r)}")

# Scale features
scaler_reward = RobustScaler()
X_train_r_scaled = scaler_reward.fit_transform(X_train_r)
X_test_r_scaled = scaler_reward.transform(X_test_r)

# Train model
xgb_reward = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=5,
    learning_rate=0.03,
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

xgb_reward.fit(X_train_r_scaled, y_train_r, verbose=False)

# Calculate metrics
from sklearn.metrics import r2_score, mean_squared_error
y_pred_r = xgb_reward.predict(X_test_r_scaled)
r2 = r2_score(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))

print(f"    R² Score: {r2:.4f}")
print(f"    RMSE: {rmse:.4f}")

# Save model
xgb_reward.save_model('xgboost_reward_predictor.json')
print(f"    Saved: xgboost_reward_predictor.json")

reward_package = {
    'model': xgb_reward,
    'scaler': scaler_reward,
    'feature_cols': feature_cols_reward,
    'metrics': {'r2_score': r2, 'rmse': rmse}
}

with open('xgboost_reward_predictor.pkl', 'wb') as f:
    pickle.dump(reward_package, f)
print(f"    Saved: xgboost_reward_predictor.pkl")

# ============================================================================
# OBJECTIVE 2B: PRICE PREDICTION MODEL
# ============================================================================
print("\n" + "="*80)
print("OBJECTIVE 2B: PRICE PREDICTION MODEL")
print("="*80)

# Feature engineering for price prediction
df_price = df.copy()
df_price['Total_supply_log'] = np.log10(df_price['Total_supply'] + 1)
df_price['Tokens_per_airdrop_log'] = np.log10(df_price['Tokens_per_airdrop'] + 1)
df_price['Days_Per_Task'] = df_price['Days_Since_Published'] / (df_price['Task_Count'] + 1)
df_price['Has_Marketing'] = ((df_price['Has_Social'] == 1) | (df_price['Has_Referral'] == 1)).astype(int)
df_price['Referral_Strength'] = df_price['Tokens_per_referral'] * df_price['Has_Referral']
df_price['Task_Engagement'] = df_price['Task_Count'] * (df_price['Has_Social'] + 1)
df_price['Supply_Reward_Ratio'] = df_price['Total_supply_log'] / (df_price['Tokens_per_airdrop_log'] + 1)
df_price['Project_Maturity'] = np.log10(df_price['Days_Since_Published'] + 1)

# Log transform target
df_price['token_price_log'] = np.log10(df_price['ICO_token_price'] + 1e-10)

feature_cols_price = [
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

X_price = df_price[feature_cols_price]
y_price = df_price['token_price_log']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_price, y_price, test_size=0.25, random_state=42
)

print(f"\n[4] Training Price Prediction Model...")
print(f"    Train: {len(X_train_p)}, Test: {len(X_test_p)}")

# Scale features
scaler_price = RobustScaler()
X_train_p_scaled = scaler_price.fit_transform(X_train_p)
X_test_p_scaled = scaler_price.transform(X_test_p)

# Train model
xgb_price = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=4,
    learning_rate=0.04,
    n_estimators=180,
    random_state=42,
    n_jobs=-1
)

xgb_price.fit(X_train_p_scaled, y_train_p, verbose=False)

# Calculate metrics
y_pred_p = xgb_price.predict(X_test_p_scaled)
r2_p = r2_score(y_test_p, y_pred_p)
rmse_p = np.sqrt(mean_squared_error(y_test_p, y_pred_p))

print(f"    R² Score: {r2_p:.4f}")
print(f"    RMSE: {rmse_p:.4f}")

# Save model
xgb_price.save_model('xgboost_price_predictor.json')
print(f"    Saved: xgboost_price_predictor.json")

price_package = {
    'model': xgb_price,
    'scaler': scaler_price,
    'feature_cols': feature_cols_price,
    'metrics': {'r2_score': r2_p, 'rmse': rmse_p}
}

with open('xgboost_price_predictor.pkl', 'wb') as f:
    pickle.dump(price_package, f)
print(f"    Saved: xgboost_price_predictor.pkl")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

print("\n[5] Testing if models can be loaded...")

try:
    # Test scam model
    test_model = xgb.XGBClassifier()
    test_model.load_model('xgboost_scam_detector.json')
    with open('xgboost_scam_detector.pkl', 'rb') as f:
        pkg = pickle.load(f)
        print(f"✓ Scam model: {len(pkg['feature_cols'])} features, scaler fitted")
    
    # Test reward model
    test_model = xgb.XGBRegressor()
    test_model.load_model('xgboost_reward_predictor.json')
    with open('xgboost_reward_predictor.pkl', 'rb') as f:
        pkg = pickle.load(f)
        print(f"✓ Reward model: {len(pkg['feature_cols'])} features, scaler fitted")
    
    # Test price model
    test_model = xgb.XGBRegressor()
    test_model.load_model('xgboost_price_predictor.json')
    with open('xgboost_price_predictor.pkl', 'rb') as f:
        pkg = pickle.load(f)
        print(f"✓ Price model: {len(pkg['feature_cols'])} features, scaler fitted")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("="*80)
    print("\nYou can now run: python3 test_predict.py")
    
except Exception as e:
    print(f"\n✗ Verification failed: {e}")

print("\n")