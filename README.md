# Stock Prediction App

A web application for stock analysis and prediction using Alpha Vantage API.

## Features

- Real-time stock data retrieval
- Moving average calculations
- Volatility analysis
- Trading simulation
- Interactive charts

## Live Demo

The app is deployed at: [Your Render URL will be here after deployment]

## Local Development

1. Clone the repository:
```bash
git clone [your-repo-url]
cd windsurf-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python app.py
```

5. Open http://localhost:5000 in your browser

## Deployment

This app is deployed on Render.com. To deploy your own instance:

1. Create a new account on [Render](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Add the following environment variables in Render:
   - `ALPHA_VANTAGE_API_KEY`: Your Alpha Vantage API key

## API Usage

The app uses the Alpha Vantage API for stock data. Get your API key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key).

## Technologies Used

- Flask
- Python
- Alpha Vantage API
- Plotly.js
- Bootstrap
- Pandas
- NumPy
