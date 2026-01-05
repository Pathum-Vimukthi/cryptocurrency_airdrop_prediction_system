#!/usr/bin/env python3
"""
Test script to verify predict_all.py works correctly
"""

import subprocess
import json
import sys

# Test data
test_data = {
    "ICO_token_price": 1.0,
    "Total_supply": 1000000000,
    "Published_Year": 2024,
    "Published_Month": 1,
    "Days_Since_Published": 365,
    "Task_Count": 5,
    "Has_Social": 1,
    "Has_Referral": 1,
    "Tokens_per_referral": 50
}

print("=" * 80)
print("Testing predict_all.py")
print("=" * 80)

print("\nTest Data:")
print(json.dumps(test_data, indent=2))

print("\n" + "=" * 80)
print("Running prediction...")
print("=" * 80 + "\n")

try:
    # Call the prediction script
    result = subprocess.run(
        ['python3', 'predict_all.py', json.dumps(test_data)],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    print("STDOUT:")
    print(result.stdout)
    
    print("\nSTDERR (Debug Info):")
    print(result.stderr)
    
    print("\nReturn Code:", result.returncode)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✓ SUCCESS!")
        print("=" * 80)
        
        # Try to parse result
        try:
            parsed = json.loads(result.stdout)
            if parsed.get('success'):
                print("\n✓ Valid JSON response")
                print(f"✓ Scam Risk: {parsed['scam_detection']['risk_level']}")
                print(f"✓ Predicted Tokens: {parsed['reward_prediction']['predicted_tokens']}")
                print(f"✓ Predicted Price: ${parsed['price_prediction']['predicted_price_usd']}")
                print(f"✓ Recommendation: {parsed['recommendation']['action']}")
            else:
                print("\n✗ Prediction failed:", parsed.get('error'))
        except json.JSONDecodeError as e:
            print("\n✗ Invalid JSON output:", e)
    else:
        print("\n" + "=" * 80)
        print("✗ FAILED!")
        print("=" * 80)
        
except subprocess.TimeoutExpired:
    print("✗ Script timeout (took longer than 30 seconds)")
except FileNotFoundError:
    print("✗ predict_all.py not found in current directory")
except Exception as e:
    print(f"✗ Test failed: {e}")

print("\n" + "=" * 80)