import React, { useState } from 'react';
import axios from 'axios';
import { 
  Shield, TrendingUp, DollarSign, AlertTriangle, 
  CheckCircle, XCircle, Info, Loader, BarChart3 
} from 'lucide-react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    ICO_token_price: 1.00,
    Total_supply: 1000000000,
    Published_Year: 2024,
    Published_Month: 1,
    Days_Since_Published: 365,
    Task_Count: 5,
    Has_Social: 1,
    Has_Referral: 1,
    Tokens_per_referral: 50
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post('http://localhost:5000/api/predict/all', formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to get prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      ICO_token_price: 1.00,
      Total_supply: 1000000000,
      Published_Year: 2024,
      Published_Month: 1,
      Days_Since_Published: 365,
      Task_Count: 5,
      Has_Social: 1,
      Has_Referral: 1,
      Tokens_per_referral: 50
    });
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <BarChart3 size={40} />
          <div>
            <h1>Cryptocurrency Airdrop Prediction System</h1>
            <p>Scam Detection, Reward & Price Prediction</p>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="form-section">
          <h2>Project Information</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>
                  <DollarSign size={16} />
                  ICO Token Price (USD)
                </label>
                <input
                  type="number"
                  name="ICO_token_price"
                  value={formData.ICO_token_price}
                  onChange={handleChange}
                  step="0.000001"
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <TrendingUp size={16} />
                  Total Supply
                </label>
                <input
                  type="number"
                  name="Total_supply"
                  value={formData.Total_supply}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Published Year
                </label>
                <input
                  type="number"
                  name="Published_Year"
                  value={formData.Published_Year}
                  onChange={handleChange}
                  min="2018"
                  max="2025"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Published Month
                </label>
                <input
                  type="number"
                  name="Published_Month"
                  value={formData.Published_Month}
                  onChange={handleChange}
                  min="1"
                  max="12"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Days Since Published
                </label>
                <input
                  type="number"
                  name="Days_Since_Published"
                  value={formData.Days_Since_Published}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Task Count
                </label>
                <input
                  type="number"
                  name="Task_Count"
                  value={formData.Task_Count}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Has Social Media
                </label>
                <select
                  name="Has_Social"
                  value={formData.Has_Social}
                  onChange={handleChange}
                  required
                >
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </div>

              <div className="form-group">
                <label>
                  <Info size={16} />
                  Has Referral Program
                </label>
                <select
                  name="Has_Referral"
                  value={formData.Has_Referral}
                  onChange={handleChange}
                  required
                >
                  <option value={1}>Yes</option>
                  <option value={0}>No</option>
                </select>
              </div>

              <div className="form-group full-width">
                <label>
                  <TrendingUp size={16} />
                  Tokens per Referral
                </label>
                <input
                  type="number"
                  name="Tokens_per_referral"
                  value={formData.Tokens_per_referral}
                  onChange={handleChange}
                  min="0"
                  required
                />
              </div>
            </div>

            <div className="button-group">
              <button type="submit" className="btn btn-primary" disabled={loading}>
                {loading ? (
                  <>
                    <Loader className="spinner" size={20} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <BarChart3 size={20} />
                    Predict
                  </>
                )}
              </button>
              <button type="button" className="btn btn-secondary" onClick={handleReset}>
                Reset
              </button>
            </div>
          </form>
        </div>

        {error && (
          <div className="error-message">
            <AlertTriangle size={20} />
            <span>{error}</span>
          </div>
        )}

        {prediction && prediction.success && (
          <div className="results-section">
            <h2>Prediction Results</h2>

            {/* Recommendation Card */}
            <div className={`recommendation-card ${prediction.recommendation.action.toLowerCase()}`}>
              <div className="recommendation-header">
                <h3>
                  {prediction.recommendation.action === 'PARTICIPATE' && <CheckCircle size={24} />}
                  {prediction.recommendation.action === 'AVOID' && <XCircle size={24} />}
                  {prediction.recommendation.action === 'CAUTION' && <AlertTriangle size={24} />}
                  {prediction.recommendation.action === 'CONSIDER' && <Info size={24} />}
                  {prediction.recommendation.action}
                </h3>
                <span className="priority-badge">{prediction.recommendation.priority}</span>
              </div>
              <p className="recommendation-reasoning">{prediction.recommendation.reasoning}</p>
              <div className="recommendation-stats">
                <div className="stat">
                  <span className="stat-label">ROI Estimate</span>
                  <span className="stat-value">{prediction.recommendation.roi_estimate}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Confidence</span>
                  <span className="stat-value">{(prediction.recommendation.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Results Grid */}
            <div className="results-grid">
              {/* Scam Detection */}
              <div className="result-card">
                <div className="card-header">
                  <Shield size={24} />
                  <h3>Scam Detection</h3>
                </div>
                <div 
                  className="risk-indicator" 
                  style={{ 
                    backgroundColor: prediction.scam_detection.risk_color,
                    padding: '12px',
                    borderRadius: '8px',
                    color: 'white',
                    fontWeight: 'bold',
                    textAlign: 'center',
                    marginBottom: '16px'
                  }}
                >
                  {prediction.scam_detection.risk_level} Risk
                </div>
                <div className="metric-row">
                  <span>Risk Score:</span>
                  <strong>{prediction.scam_detection.risk_score}/100</strong>
                </div>
                <div className="metric-row">
                  <span>Scam Probability:</span>
                  <strong>{(prediction.scam_detection.scam_probability * 100).toFixed(2)}%</strong>
                </div>
                <div className="metric-row">
                  <span>Status:</span>
                  <strong>{prediction.scam_detection.is_scam ? 'Potential Scam' : 'Legitimate'}</strong>
                </div>
                
                {Object.values(prediction.scam_detection.warning_flags).some(flag => flag) && (
                  <div className="warning-flags">
                    <h4>⚠️ Warning Flags:</h4>
                    <ul>
                      {prediction.scam_detection.warning_flags.high_reward_ratio && (
                        <li>High reward ratio (pyramid scheme indicator)</li>
                      )}
                      {prediction.scam_detection.warning_flags.high_inflation_risk && (
                        <li>High supply inflation risk</li>
                      )}
                      {prediction.scam_detection.warning_flags.very_new_project && (
                        <li>Very new project (less than 6 months)</li>
                      )}
                      {prediction.scam_detection.warning_flags.no_social_media && (
                        <li>No social media presence</li>
                      )}
                      {prediction.scam_detection.warning_flags.suspicious_tasks && (
                        <li>Suspicious task structure</li>
                      )}
                    </ul>
                  </div>
                )}
              </div>

              {/* Reward Prediction */}
              <div className="result-card">
                <div className="card-header">
                  <TrendingUp size={24} />
                  <h3>Reward Prediction</h3>
                </div>
                <div className="metric-row highlight">
                  <span>Predicted Tokens:</span>
                  <strong>{prediction.reward_prediction.predicted_tokens.toLocaleString()}</strong>
                </div>
                <div className="metric-row">
                  <span>Referral Bonus:</span>
                  <strong>{prediction.reward_prediction.referral_bonus.toLocaleString()}</strong>
                </div>
                <div className="metric-row highlight">
                  <span>Total Potential:</span>
                  <strong>{prediction.reward_prediction.total_earning_potential.toLocaleString()}</strong>
                </div>
                <div className="metric-row">
                  <span>Earning Efficiency:</span>
                  <strong>{prediction.reward_prediction.earning_efficiency.toFixed(2)}</strong>
                </div>
                <div className="metric-row">
                  <span>Tasks Required:</span>
                  <strong>{prediction.reward_prediction.tasks_required}</strong>
                </div>
                <div className="metric-row">
                  <span>Tokens per Task:</span>
                  <strong>{prediction.reward_prediction.tokens_per_task.toFixed(2)}</strong>
                </div>
              </div>

              {/* Price Prediction */}
              <div className="result-card">
                <div className="card-header">
                  <DollarSign size={24} />
                  <h3>Price Prediction</h3>
                </div>
                <div className="metric-row highlight">
                  <span>Predicted Price:</span>
                  <strong>${prediction.price_prediction.predicted_price_usd.toFixed(6)}</strong>
                </div>
                <div className="metric-row">
                  <span>ICO Price:</span>
                  <strong>${prediction.price_prediction.ico_price_usd.toFixed(6)}</strong>
                </div>
                <div className="metric-row">
                  <span>Price Change:</span>
                  <strong className={prediction.price_prediction.price_change_percent >= 0 ? 'positive' : 'negative'}>
                    {prediction.price_prediction.price_change_percent >= 0 ? '+' : ''}
                    {prediction.price_prediction.price_change_percent.toFixed(2)}%
                  </strong>
                </div>
                <div className="metric-row">
                  <span>Market Cap:</span>
                  <strong>${prediction.price_prediction.market_cap_estimate.toLocaleString()}</strong>
                </div>
                <div className="metric-row">
                  <span>Stability Score:</span>
                  <strong>{prediction.price_prediction.stability_score}</strong>
                </div>
                <div className="metric-row">
                  <span>Trend:</span>
                  <strong className={prediction.price_prediction.price_trend === 'Bullish' ? 'positive' : 'negative'}>
                    {prediction.price_prediction.price_trend}
                  </strong>
                </div>
              </div>

              {/* Investment Summary */}
              <div className="result-card investment-summary">
                <div className="card-header">
                  <DollarSign size={24} />
                  <h3>Investment Summary</h3>
                </div>
                <div className="metric-row highlight">
                  <span>Expected Value:</span>
                  <strong>${prediction.investment_summary.expected_value_usd.toFixed(2)}</strong>
                </div>
                <div className="metric-row">
                  <span>ROI Estimate:</span>
                  <strong>{prediction.investment_summary.roi_estimate}</strong>
                </div>
                <div className="metric-row">
                  <span>Time Investment:</span>
                  <strong>{prediction.investment_summary.time_investment}</strong>
                </div>
                <div className="metric-row">
                  <span>Value per Task:</span>
                  <strong>${prediction.investment_summary.earning_per_task.toFixed(2)}</strong>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="footer">
        <p>Powered by XGBoost Machine Learning Models | Accuracy: 99.33%+ | R² Score: 1.0000</p>
      </footer>
    </div>
  );
}

export default App;
