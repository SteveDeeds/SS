"""
Market data downloader using yfinance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import csv
import os


def get_symbol_data(symbol: str, period: str = "4y", force_download: bool = False) -> List[Dict]:
    """
    Get historical data for a symbol, using cached data if available
    
    Args:
        symbol: Stock symbol (e.g., 'SPXL')
        period: Time period ('1y', '2y', '4y', '5y', etc.)
        force_download: If True, always download fresh data from Yahoo
        
    Returns:
        List of OHLCV dictionaries
    """
    # Check for cached data first (unless forced to download)
    if not force_download:
        cached_data = load_symbol_data(symbol)
        if cached_data:
            # Check if cached data is sufficient for the requested period
            if is_cached_data_sufficient(cached_data, period):
                # print(f"✅ Using cached data for {symbol} ({len(cached_data)} days)")
                return filter_data_by_period(cached_data, period)
            else:
                print(f"⚠️  Cached data for {symbol} is insufficient for {period} period")
    
    # Download fresh data if no cache or cache is insufficient
    return download_symbol_data(symbol, period, save_to_file=True)


def download_symbol_data(symbol: str, period: str = "4y", save_to_file: bool = True) -> List[Dict]:
    """
    Download historical data for a symbol from Yahoo Finance
    
    Args:
        symbol: Stock symbol (e.g., 'SPXL')
        period: Time period ('1y', '2y', '5y', etc.)
        save_to_file: Whether to save data to CSV file
        
    Returns:
        List of OHLCV dictionaries
    """
    try:
        print(f"📡 Downloading fresh data for {symbol} from Yahoo Finance...")
        # Download data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Convert to list of dictionaries
        price_history = []
        for date, row in df.iterrows():
            # Convert pandas datetime to datetime with market close time (4:00 PM)
            market_datetime = date.to_pydatetime().replace(hour=16, minute=0, second=0, microsecond=0)
            
            price_data = {
                'date': market_datetime,  # Full datetime object with market close time
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': int(row['Volume']),
                'symbol': symbol
            }
            price_history.append(price_data)
        
        if save_to_file:
            # Save to CSV file - fix the path calculation
            current_dir = os.path.dirname(__file__)  # src/data directory
            src_dir = os.path.dirname(current_dir)   # src directory
            project_root = os.path.dirname(src_dir)  # SS directory
            data_dir = os.path.join(project_root, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            csv_file = os.path.join(data_dir, f"{symbol}.csv")
            
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for record in price_history:
                    # Convert datetime to string for CSV
                    record_copy = record.copy()
                    record_copy['date'] = record['date'].strftime('%Y-%m-%d')
                    writer.writerow(record_copy)
            
            print(f"Data saved to {csv_file}")
        
        print(f"Downloaded {len(price_history)} days of data for {symbol}")
        return price_history
        
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")
        return []


def is_cached_data_sufficient(cached_data: List[Dict], period: str) -> bool:
    """
    Check if cached data covers the requested period
    
    Args:
        cached_data: List of OHLCV dictionaries
        period: Requested period (e.g., '1y', '2y', '4y')
        
    Returns:
        True if cached data is sufficient
    """
    if not cached_data:
        return False
    
    # Parse period string
    period_mapping = {
        '1y': 365,
        '2y': 730,
        '3y': 1095,
        '4y': 1460,
        '5y': 1825,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        'ytd': 365,  # Approximate
        'max': 0  # Special case - use all available data
    }
    
    required_days = period_mapping.get(period.lower(), 1460)  # Default to 4 years
    
    # Special case for 'max' - always sufficient
    if period.lower() == 'max':
        return True
    
    # Check if we have enough data (flexible for real-world data availability)
    available_days = len(cached_data)
    
    # For trading data, accept reasonable minimums:
    # - For 2y request (730 days), accept anything over 400 days (about 1.5 years)
    # - For 1y request (365 days), accept anything over 250 days (about 10 months)
    if required_days >= 700:  # 2+ year request
        min_required_days = max(400, int(required_days * 0.6))
    elif required_days >= 300:  # 1+ year request
        min_required_days = max(250, int(required_days * 0.7))
    else:  # Shorter periods
        min_required_days = int(required_days * 0.8)
    
    # Also check if data is reasonably recent (within last 30 days for flexibility)
    latest_date = cached_data[-1]['date']
    # Keep as datetime for consistent comparison
    if not isinstance(latest_date, datetime):
        # Parse string to datetime and add market close time
        date_obj = datetime.strptime(str(latest_date), '%Y-%m-%d')
        latest_date = date_obj.replace(hour=16, minute=0, second=0, microsecond=0)
    
    today = datetime.now()
    days_old = (today - latest_date).days
    
    # Debug logging for cache decisions
    # print(f"   Cache validation: have {available_days} days, need {min_required_days}, data age {days_old} days")
    
    # Data is sufficient if we have minimum required days and it's reasonably recent
    return available_days >= min_required_days and days_old <= 30


def filter_data_by_period(data: List[Dict], period: str) -> List[Dict]:
    """
    Filter cached data to match the requested period
    
    Args:
        data: Full cached data
        period: Requested period
        
    Returns:
        Filtered data for the requested period
    """
    if period.lower() == 'max':
        return data
    
    # Parse period to days
    period_mapping = {
        '1y': 365,
        '2y': 730,
        '3y': 1095,
        '4y': 1460,
        '5y': 1825,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        'ytd': 365,  # Approximate
    }
    
    required_days = period_mapping.get(period.lower(), 1460)
    
    # Return the most recent N days
    if len(data) <= required_days:
        return data
    else:
        return data[-required_days:]


def load_symbol_data(symbol: str, data_dir: str = None) -> List[Dict]:
    """
    Load historical data from CSV file
    
    Args:
        symbol: Stock symbol
        data_dir: Directory containing data files
        
    Returns:
        List of OHLCV dictionaries
    """
    if data_dir is None:
        current_dir = os.path.dirname(__file__)  # src/data directory
        src_dir = os.path.dirname(current_dir)   # src directory  
        project_root = os.path.dirname(src_dir)  # SS directory
        data_dir = os.path.join(project_root, 'data')
    
    csv_file = os.path.join(data_dir, f"{symbol}.csv")
    
    if not os.path.exists(csv_file):
        print(f"Data file not found: {csv_file}")
        return []
    
    price_history = []
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse date string and add market close time (4:00 PM)
                date_obj = datetime.strptime(row['date'], '%Y-%m-%d')
                market_datetime = date_obj.replace(hour=16, minute=0, second=0, microsecond=0)
                
                price_data = {
                    'date': market_datetime,  # Full datetime object with market close time
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume']),
                    'symbol': row['symbol']
                }
                price_history.append(price_data)
        
        # print(f"Loaded {len(price_history)} days of data for {symbol}")
        return price_history
        
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        return []


if __name__ == "__main__":
    # Test the new caching functionality
    print("Testing data loading with caching...")
    spxl_data = get_symbol_data("SPXL", period="2y")
    if spxl_data:
        print(f"First record: {spxl_data[0]}")
        print(f"Last record: {spxl_data[-1]}")
        
    # Test force download
    print("\nTesting force download...")
    spxl_data_fresh = get_symbol_data("SPXL", period="2y", force_download=True)
    if spxl_data_fresh:
        print(f"Fresh data length: {len(spxl_data_fresh)}")
