from flask import Flask, jsonify, request, render_template
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
BASE_URL = 'https://www.alphavantage.co/query'

def clean_numeric_data(df):
    """Clean numeric data by replacing NaN/inf values with None"""
    return df.replace([np.inf, -np.inf, np.nan], None)

def get_stock_data(symbol, function='TIME_SERIES_DAILY'):
    """
    Fetch stock data from Alpha Vantage API
    """
    logger.info(f"Fetching {function} data for symbol: {symbol}")
    
    params = {
        'function': function,
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    
    try:
        logger.debug(f"Making request to Alpha Vantage with params: {params}")
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Log the complete response for debugging
        logger.debug(f"Complete API Response: {data}")
        
        # Check for error messages
        if "Error Message" in data:
            logger.error(f"API Error: {data['Error Message']}")
            return None, data["Error Message"]
            
        if "Note" in data:  # API limit reached
            logger.warning(f"API Note: {data['Note']}")
            return None, "API rate limit reached. Please try again later."
            
        # For TIME_SERIES_DAILY, check both adjusted and regular endpoints
        time_series_keys = [
            'Time Series (Daily)',
            'Weekly Time Series',
            'Monthly Time Series'
        ]
        
        time_series_key = None
        for key in time_series_keys:
            if key in data:
                time_series_key = key
                break
                
        if not time_series_key:
            logger.error(f"No time series data found. Available keys: {list(data.keys())}")
            return None, "No time series data available. Please try a different symbol."
            
        logger.info(f"Successfully found time series data with key: {time_series_key}")
        return data, None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return None, f"Error fetching data: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None, f"Unexpected error: {str(e)}"

def parse_time_series(data):
    """
    Parse the time series data from Alpha Vantage response
    """
    try:
        # Log the input data structure
        logger.debug(f"Parsing time series data with keys: {list(data.keys())}")
        
        # Find the time series key in the response
        time_series_keys = [
            'Time Series (Daily)',
            'Weekly Time Series',
            'Monthly Time Series'
        ]
        
        time_series_key = None
        for key in time_series_keys:
            if key in data:
                time_series_key = key
                break
        
        if not time_series_key:
            logger.error("No valid time series key found in data")
            return pd.DataFrame()
        
        logger.info(f"Using time series key: {time_series_key}")
        
        # Get the time series data
        time_series_data = data[time_series_key]
        logger.debug(f"First data point: {next(iter(time_series_data.items()))}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series_data, orient='index')
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Clean numeric data
        for col in df.columns:
            # Remove currency symbols and commas
            df[col] = df[col].str.replace('$', '').str.replace(',', '')
            # Convert to numeric, setting errors to coerce will convert invalid values to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename columns
        column_mapping = {
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        }
        
        # Only rename columns that exist
        rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_cols)
        
        # Sort by date
        df = df.sort_index()
        
        # Clean any NaN or infinite values
        df = clean_numeric_data(df)
        
        logger.info(f"Successfully parsed data. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing time series data: {str(e)}")
        return pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stocks', methods=['GET'])
def get_stock_info():
    symbol = request.args.get('symbol', '').strip().upper()
    logger.info(f"Stock data requested for symbol: {symbol}")
    
    if not symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    # Get daily data
    data, error = get_stock_data(symbol)
    if error:
        return jsonify({'error': error}), 400
    
    # Parse the time series data
    df = parse_time_series(data)
    if df.empty:
        return jsonify({'error': 'No data available for this symbol'}), 400
    
    # Get the latest price
    latest_price = df['Close'].iloc[-1]
    if pd.isna(latest_price):
        latest_price = None
    else:
        latest_price = float(latest_price)
    
    # Clean the data before converting to dict
    df = clean_numeric_data(df)
    
    # Convert timestamps to strings for JSON serialization
    result = {
        'current_price': latest_price,
        'historical_data': {k.strftime('%Y-%m-%d'): v for k, v in df.to_dict('index').items()}
    }
    
    return jsonify(result)

@app.route('/api/moving-average', methods=['GET'])
def moving_average():
    symbol = request.args.get('symbol', '').strip().upper()
    logger.info(f"Moving average requested for symbol: {symbol}")
    
    try:
        window = int(request.args.get('window', 20))
        if window < 1:
            return jsonify({'error': 'Window must be positive'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid window value'}), 400
    
    # Get daily data
    data, error = get_stock_data(symbol)
    if error:
        return jsonify({'error': error}), 400
    
    # Parse and calculate moving average
    df = parse_time_series(data)
    if df.empty:
        return jsonify({'error': 'No data available for this symbol'}), 400
    
    # Calculate moving average
    df['Moving_Average'] = df['Close'].rolling(window=window).mean()
    
    # Clean the data before converting to dict
    df = clean_numeric_data(df)
    
    # Convert timestamps to strings for JSON serialization
    result = {
        'Close': {k.strftime('%Y-%m-%d'): v for k, v in df['Close'].to_dict().items()},
        'Moving_Average': {k.strftime('%Y-%m-%d'): v for k, v in df['Moving_Average'].to_dict().items()}
    }
    
    return jsonify(result)

@app.route('/api/volatility', methods=['GET'])
def volatility():
    symbol = request.args.get('symbol', '').strip().upper()
    logger.info(f"Volatility requested for symbol: {symbol}")
    
    # Get daily data
    data, error = get_stock_data(symbol)
    if error:
        return jsonify({'error': error}), 400
    
    # Parse and calculate volatility
    df = parse_time_series(data)
    if df.empty:
        return jsonify({'error': 'No data available for this symbol'}), 400
    
    # Clean the data
    df = clean_numeric_data(df)
    
    # Calculate daily returns
    returns = df['Close'].pct_change().dropna()
    
    # Calculate annualized volatility (252 trading days in a year)
    volatility = returns.std() * np.sqrt(252)
    
    # Calculate 95% confidence interval for returns
    confidence_interval = (
        returns.mean() - 1.96 * returns.std(),
        returns.mean() + 1.96 * returns.std()
    )
    
    # Clean any NaN values
    if pd.isna(volatility):
        volatility = None
    else:
        volatility = float(volatility)
    
    confidence_interval = [
        None if pd.isna(x) else float(x)
        for x in confidence_interval
    ]
    
    result = {
        'volatility': volatility,
        'confidence_interval': confidence_interval
    }
    
    return jsonify(result)

@app.route('/api/trade', methods=['POST'])
def trade():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        action = data.get('action', '').lower()
        symbol = data.get('symbol', '').strip().upper()
        quantity = data.get('quantity')
        
        logger.info(f"Trade requested: {action} {quantity} shares of {symbol}")
        
        # Validate inputs
        if not all([action, symbol, quantity]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if action not in ['buy', 'sell']:
            return jsonify({'error': 'Invalid action. Must be "buy" or "sell"'}), 400
        
        try:
            quantity = int(quantity)
            if quantity <= 0:
                return jsonify({'error': 'Quantity must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid quantity value'}), 400
        
        # Get current price
        stock_data, error = get_stock_data(symbol)
        if error:
            return jsonify({'error': error}), 400
        
        df = parse_time_series(stock_data)
        if df.empty:
            return jsonify({'error': 'Unable to get current price'}), 400
        
        # Clean the data
        df = clean_numeric_data(df)
        
        current_price = df['Close'].iloc[-1]
        if pd.isna(current_price):
            return jsonify({'error': 'Unable to determine current price'}), 400
            
        current_price = float(current_price)
        total_value = current_price * quantity
        
        return jsonify({
            'message': f'{action.capitalize()} order for {quantity} shares of {symbol} executed successfully',
            'details': {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price_per_share': current_price,
                'total_value': total_value,
                'status': 'success'
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing trade: {str(e)}")
        return jsonify({'error': f'Error processing trade: {str(e)}'}), 400

if __name__ == '__main__':
    # Only bind to localhost when running in development
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
