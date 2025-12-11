const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Airdrop Prediction API is running',
    timestamp: new Date().toISOString()
  });
});

// Scam Detection Endpoint
app.post('/api/predict/scam', (req, res) => {
  const inputData = req.body;
  
  // Validate required fields
  const requiredFields = [
    'ICO_token_price', 'Total_supply', 'Published_Year', 'Published_Month',
    'Days_Since_Published', 'Task_Count', 'Has_Social', 'Has_Referral',
    'Tokens_per_referral'
  ];
  
  const missingFields = requiredFields.filter(field => !(field in inputData));
  if (missingFields.length > 0) {
    return res.status(400).json({
      error: 'Missing required fields',
      missingFields: missingFields
    });
  }

  // Call Python prediction script
  const python = spawn('python3', [
    path.join(__dirname, 'predict_scam.py'),
    JSON.stringify(inputData)
  ]);

  let dataString = '';
  let errorString = '';

  python.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  python.stderr.on('data', (data) => {
    errorString += data.toString();
  });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error('Python Error:', errorString);
      return res.status(500).json({
        error: 'Prediction failed',
        details: errorString
      });
    }

    try {
      const result = JSON.parse(dataString);
      res.json(result);
    } catch (e) {
      console.error('Parse Error:', e);
      console.error('Raw output:', dataString);
      res.status(500).json({
        error: 'Failed to parse prediction result',
        details: e.message
      });
    }
  });
});

// Reward Prediction Endpoint
app.post('/api/predict/reward', (req, res) => {
  const inputData = req.body;
  
  // Validate required fields
  const requiredFields = [
    'Total_supply', 'Tokens_per_referral', 'Task_Count',
    'Has_Referral', 'Has_Social', 'Published_Year', 'Days_Since_Published'
  ];
  
  const missingFields = requiredFields.filter(field => !(field in inputData));
  if (missingFields.length > 0) {
    return res.status(400).json({
      error: 'Missing required fields',
      missingFields: missingFields
    });
  }

  // Call Python prediction script
  const python = spawn('python3', [
    path.join(__dirname, 'predict_reward.py'),
    JSON.stringify(inputData)
  ]);

  let dataString = '';
  let errorString = '';

  python.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  python.stderr.on('data', (data) => {
    errorString += data.toString();
  });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error('Python Error:', errorString);
      return res.status(500).json({
        error: 'Prediction failed',
        details: errorString
      });
    }

    try {
      const result = JSON.parse(dataString);
      res.json(result);
    } catch (e) {
      console.error('Parse Error:', e);
      res.status(500).json({
        error: 'Failed to parse prediction result',
        details: e.message
      });
    }
  });
});

// Price Prediction Endpoint
app.post('/api/predict/price', (req, res) => {
  const inputData = req.body;
  
  // Validate required fields
  const requiredFields = [
    'Total_supply', 'Tokens_per_referral', 'Task_Count',
    'Has_Referral', 'Has_Social', 'Published_Year', 'Days_Since_Published'
  ];
  
  const missingFields = requiredFields.filter(field => !(field in inputData));
  if (missingFields.length > 0) {
    return res.status(400).json({
      error: 'Missing required fields',
      missingFields: missingFields
    });
  }

  // Call Python prediction script
  const python = spawn('python3', [
    path.join(__dirname, 'predict_price.py'),
    JSON.stringify(inputData)
  ]);

  let dataString = '';
  let errorString = '';

  python.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  python.stderr.on('data', (data) => {
    errorString += data.toString();
  });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error('Python Error:', errorString);
      return res.status(500).json({
        error: 'Prediction failed',
        details: errorString
      });
    }

    try {
      const result = JSON.parse(dataString);
      res.json(result);
    } catch (e) {
      console.error('Parse Error:', e);
      res.status(500).json({
        error: 'Failed to parse prediction result',
        details: e.message
      });
    }
  });
});

// Combined Prediction Endpoint (All 3 models)
app.post('/api/predict/all', (req, res) => {
  const inputData = req.body;
  
  // Validate all required fields
  const requiredFields = [
    'ICO_token_price', 'Total_supply', 'Published_Year', 'Published_Month',
    'Days_Since_Published', 'Task_Count', 'Has_Social', 'Has_Referral',
    'Tokens_per_referral'
  ];
  
  const missingFields = requiredFields.filter(field => !(field in inputData));
  if (missingFields.length > 0) {
    return res.status(400).json({
      error: 'Missing required fields',
      missingFields: missingFields
    });
  }

  // Call Python prediction script for all models
  const python = spawn('python3', [
    path.join(__dirname, 'predict_all.py'),
    JSON.stringify(inputData)
  ]);

  let dataString = '';
  let errorString = '';

  python.stdout.on('data', (data) => {
    dataString += data.toString();
  });

  python.stderr.on('data', (data) => {
    errorString += data.toString();
  });

  python.on('close', (code) => {
    if (code !== 0) {
      console.error('Python Error:', errorString);
      return res.status(500).json({
        error: 'Prediction failed',
        details: errorString
      });
    }

    try {
      const result = JSON.parse(dataString);
      res.json(result);
    } catch (e) {
      console.error('Parse Error:', e);
      console.error('Raw output:', dataString);
      res.status(500).json({
        error: 'Failed to parse prediction result',
        details: e.message
      });
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server Error:', err);
  res.status(500).json({
    error: 'Internal server error',
    message: err.message
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Airdrop Prediction API running on http://localhost:${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/api/health`);
});
