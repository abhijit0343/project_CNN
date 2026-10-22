// config/index.js
// Central configuration loaded from environment variables.
// Import this wherever you need app-level settings.

require('dotenv').config();

const config = {
  port    : process.env.PORT || 5000,
  nodeEnv : process.env.NODE_ENV || 'development',

  // Python model service base URL
  modelServiceUrl: process.env.MODEL_SERVICE_URL || 'http://localhost:8000',
};

module.exports = config;
