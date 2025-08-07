"""
Data Augmentation Module for generating synthetic market scenarios
using Bootstrap sampling with Monte Carlo noise overlay
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import copy


class BootstrapNoiseGenerator:
    """
    Generate synthetic price paths using bootstrap sampling with Monte Carlo noise overlay
    """
    
    def __init__(self):
        self.random_state = np.random.RandomState()  # New random seed each instance
    
    def generate_bootstrap_with_noise_scenarios(self, 
                                               historical_data: List[Dict], 
                                               n_scenarios: int = 100,
                                               scenario_length: int = 252,
                                               market_regime: str = 'all',
                                               block_size: int = 20,
                                               noise_level: float = 1.0,
                                               return_adjustment: float = 0.0,
                                               correlation_preservation: bool = True) -> List[List[Dict]]:
        """
        Generate scenarios using bootstrap sampling with noise overlay
        
        Args:
            historical_data: Original OHLCV data (list of dicts)
            n_scenarios: Number of synthetic scenarios to generate
            scenario_length: Length of each scenario (trading days)
            market_regime: 'all', 'bull_only', 'bear_only', 'sideways_only', 'mixed'
            block_size: Size of blocks for block bootstrap (preserves autocorrelation)
            noise_level: Monte Carlo noise scaling (0.0 = pure bootstrap, 2.0 = double noise)
            return_adjustment: Adjust mean returns (+/- percentage points)
            correlation_preservation: Whether to maintain cross-asset correlations
            
        Returns:
            List of synthetic OHLCV datasets with regime and noise metadata
        """
        # print(f"Generating {n_scenarios} scenarios with {scenario_length} days each...")
        # print(f"Market regime: {market_regime}, Block size: {block_size}, Noise level: {noise_level}")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(historical_data)
        # Ensure datetime consistency - convert to pandas datetime then preserve as datetime objects
        df['date'] = pd.to_datetime(df['date'])  # Keep as datetime objects throughout
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        df = df.dropna().reset_index(drop=True)
        
        # Filter historical blocks by regime if specified
        filtered_data = self._filter_historical_blocks_by_regime(df, market_regime)
        
        # Calibrate noise parameters for the regime
        regime_params = self._calibrate_regime_noise_parameters(filtered_data, market_regime)
        
        scenarios = []
        
        for scenario_idx in range(n_scenarios):
            # if scenario_idx % 50 == 0:
            #     print(f"Generated {scenario_idx}/{n_scenarios} scenarios...")
            
            # Generate one synthetic scenario
            synthetic_scenario = self._generate_single_scenario(
                filtered_data, scenario_length, block_size, 
                noise_level, return_adjustment, regime_params
            )
            
            scenarios.append(synthetic_scenario)
        
        # print(f"Successfully generated {len(scenarios)} scenarios")
        return scenarios
    
    def _filter_historical_blocks_by_regime(self, df: pd.DataFrame, market_regime: str) -> pd.DataFrame:
        """
        Filter historical data blocks based on market regime
        """
        if market_regime == 'all':
            return df
        
        # Calculate rolling returns for regime classification
        rolling_window = 20  # 4-week periods
        df['rolling_return'] = df['return'].rolling(window=rolling_window).sum()
        
        if market_regime == 'bull_only':
            # Periods with positive rolling returns
            mask = df['rolling_return'] > 0.02  # > 2% over 20 days
        elif market_regime == 'bear_only':
            # Periods with negative rolling returns
            mask = df['rolling_return'] < -0.02  # < -2% over 20 days
        elif market_regime == 'sideways_only':
            # Periods with small rolling returns
            mask = (df['rolling_return'] >= -0.02) & (df['rolling_return'] <= 0.02)
        else:
            # Default to all data
            mask = pd.Series([True] * len(df))
        
        filtered_df = df[mask].copy()
        
        if len(filtered_df) < 50:  # Minimum viable data
            print(f"Warning: Only {len(filtered_df)} days match regime '{market_regime}', using all data")
            return df
        
        print(f"Filtered to {len(filtered_df)} days for regime '{market_regime}'")
        return filtered_df
    
    def _calibrate_regime_noise_parameters(self, df: pd.DataFrame, regime: str) -> Dict:
        """
        Calibrate noise parameters based on historical regime characteristics
        """
        returns = df['return'].dropna()
        
        params = {
            'volatility': returns.std(),
            'mean_return': returns.mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'autocorr_1': returns.autocorr(lag=1) if len(returns) > 1 else 0.0
        }
        
        # Adjust parameters based on regime
        if regime == 'bear_only':
            params['volatility'] *= 1.3  # Higher volatility in bear markets
            params['mean_return'] *= 1.1  # More negative returns
        elif regime == 'bull_only':
            params['volatility'] *= 0.9  # Lower volatility in bull markets
            params['mean_return'] *= 1.1  # More positive returns
        
        return params
    
    def _generate_single_scenario(self, df: pd.DataFrame, scenario_length: int, 
                                 block_size: int, noise_level: float, 
                                 return_adjustment: float, regime_params: Dict) -> List[Dict]:
        """
        Generate a single synthetic scenario using block bootstrap with noise
        """
        # Start with a random historical day as base
        index = self.random_state.randint(0, len(df))
        start_price = df.iloc[index]['close']
        start_date = df.iloc[index]['date']  # Use datetime instead of date
        
        # Generate bootstrap returns
        bootstrap_returns = self._block_bootstrap_returns(df, scenario_length, block_size)
        
        # Apply Monte Carlo noise
        if noise_level > 0:
            bootstrap_returns = self._apply_monte_carlo_noise(
                bootstrap_returns, noise_level, regime_params
            )
        
        # Apply return adjustment
        if return_adjustment != 0:
            bootstrap_returns += return_adjustment / 252  # Daily adjustment
        
        # Generate OHLCV data from returns with historical OHLCV structure
        scenario = self._generate_ohlcv_from_returns(
            bootstrap_returns, start_price, start_date, df.iloc[0]['symbol'], df, noise_level
        )
        
        # Validate OHLC constraints
        scenario = self._validate_ohlc_constraints(scenario)
        
        return scenario
    
    def _block_bootstrap_returns(self, df: pd.DataFrame, scenario_length: int, block_size: int) -> np.ndarray:
        """
        Perform block bootstrap sampling of returns
        """
        returns = df['return'].dropna().values
        
        if len(returns) < block_size:
            block_size = max(1, len(returns) // 2)
        
        bootstrap_returns = []
        
        while len(bootstrap_returns) < scenario_length:
            # Randomly select a block start position
            max_start = len(returns) - block_size
            if max_start <= 0:
                # Fallback to single observation sampling
                start_idx = self.random_state.randint(0, len(returns))
                block = [returns[start_idx]]
            else:
                start_idx = self.random_state.randint(0, max_start + 1)
                block = returns[start_idx:start_idx + block_size]
            
            bootstrap_returns.extend(block)
        
        # Trim to exact length
        return np.array(bootstrap_returns[:scenario_length])
    
    def _apply_monte_carlo_noise(self, bootstrap_returns: np.ndarray, 
                                noise_level: float, regime_params: Dict) -> np.ndarray:
        """
        Apply Monte Carlo noise to bootstrap-sampled returns
        """
        n_returns = len(bootstrap_returns)
        
        # Generate noise based on regime parameters
        noise_std = regime_params['volatility'] * noise_level
        noise = self.random_state.normal(0, noise_std, n_returns)
        
        # Apply autocorrelation structure if present
        if regime_params.get('autocorr_1', 0) != 0:
            autocorr = regime_params['autocorr_1']
            for i in range(1, n_returns):
                noise[i] += autocorr * noise[i-1] * 0.3  # Damped autocorrelation
        
        # Combine bootstrap returns with noise
        augmented_returns = bootstrap_returns + noise
        
        return augmented_returns
    
    def _generate_ohlcv_from_returns(self, returns: np.ndarray, start_price: float, 
                                   start_date: datetime, symbol: str, historical_df: pd.DataFrame,
                                   noise_level: float = 0.005) -> List[Dict]:
        """
        Generate OHLCV data from return series
        """
        scenario = []
        current_price = start_price
        
        for i, daily_return in enumerate(returns):
            # Calculate new close price from bootstrap return
            new_close = current_price * (1 + daily_return)
            
            # Find the corresponding historical day for this bootstrap sample
            # We'll use the return to match back to historical OHLCV structure
            historical_day_idx = i % len(historical_df)  # Cycle through historical data
            historical_day = historical_df.iloc[historical_day_idx]
            
            # Extract historical OHLC ratios relative to close
            hist_open_ratio = historical_day['open'] / historical_day['close'] if historical_day['close'] != 0 else 1.0
            hist_high_ratio = historical_day['high'] / historical_day['close'] if historical_day['close'] != 0 else 1.0
            hist_low_ratio = historical_day['low'] / historical_day['close'] if historical_day['close'] != 0 else 1.0
            
            # Apply historical ratios to new close price with configurable noise
            noise_pct = noise_level * 0.01  # Convert noise_level to percentage (e.g., 1.0 -> 1%)
            
            # Generate base OHLC from historical ratios
            base_open = new_close * hist_open_ratio
            base_high = new_close * hist_high_ratio  
            base_low = new_close * hist_low_ratio
            base_close = new_close
            
            # Apply noise while maintaining OHLC constraints
            # Start with close (no noise on close to maintain return accuracy)
            close_price = base_close
            
            # Generate open with noise, but ensure it's within reasonable bounds
            open_noise = self.random_state.normal(0, noise_pct)
            open_price = base_open * (1 + open_noise)
            
            # Generate high and low with noise, ensuring they bracket open and close
            high_noise = self.random_state.normal(0, noise_pct)
            low_noise = self.random_state.normal(0, noise_pct)
            
            # Calculate noisy high and low
            noisy_high = base_high * (1 + high_noise)
            noisy_low = base_low * (1 + low_noise)
            
            # Ensure OHLC constraints: High >= max(O,L,C) and Low <= min(O,L,C)
            min_ohlc = min(open_price, close_price)
            max_ohlc = max(open_price, close_price)
            
            high_price = max(noisy_high, max_ohlc)  # High must be at least the max of O,L,C
            low_price = min(noisy_low, min_ohlc)    # Low must be at most the min of O,H,C
            
            # Ensure High > Low (minimum spread)
            if high_price <= low_price:
                spread = abs(close_price * 0.001)  # Minimum 0.1% spread
                high_price = low_price + spread
            
            # Use historical volume with noise
            historical_volume = historical_day['volume']
            volume = int(historical_volume * (1 + self.random_state.normal(0, noise_pct * 2)))  # Double noise for volume
            volume = max(1, volume)  # Ensure positive volume
            
            day_data = {
                'date': start_date + timedelta(days=i),  # This will be a datetime object
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': volume,
                'symbol': symbol
            }
            
            scenario.append(day_data)
            current_price = close_price
        
        return scenario
    
    def _validate_ohlc_constraints(self, scenario: List[Dict]) -> List[Dict]:
        """
        Validate that generated OHLC data meets required constraints
        """
        for day_data in scenario:
            # Ensure High >= Open, Low, Close
            day_data['high'] = max(day_data['high'], day_data['open'], 
                                 day_data['low'], day_data['close'])
            
            # Ensure Low <= Open, High, Close
            day_data['low'] = min(day_data['low'], day_data['open'], 
                                day_data['high'], day_data['close'])
            
            # Ensure High > Low
            if day_data['high'] <= day_data['low']:
                day_data['high'] = day_data['low'] * 1.001  # Minimum 0.1% spread
            
            # Ensure positive volume
            day_data['volume'] = max(1, day_data['volume'])
        
        return scenario


class DataAugmentationEngine:
    """
    Main engine for generating augmented market data
    """
    
    def __init__(self):
        self.bootstrap_noise_generator = BootstrapNoiseGenerator()
    
    def generate_augmented_data(self, historical_data: List[Dict], 
                              n_scenarios: int = 100,
                              scenario_length: int = 252,
                              method: str = 'bootstrap_with_noise',
                              **kwargs) -> List[List[Dict]]:
        """
        Generate augmented price histories based on configuration
        
        Args:
            historical_data: Original OHLCV data
            n_scenarios: Number of scenarios to generate
            scenario_length: Length of each scenario in days
            method: Augmentation method ('bootstrap_with_noise')
            **kwargs: Additional parameters for the specific method
            
        Returns:
            List of synthetic OHLCV datasets
        """
        if method == 'bootstrap_with_noise':
            return self.bootstrap_noise_generator.generate_bootstrap_with_noise_scenarios(
                historical_data, n_scenarios, scenario_length, **kwargs
            )
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
