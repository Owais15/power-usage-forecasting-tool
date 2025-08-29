import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Data Processing class for handling power usage data
    Handles data cleaning, validation, and preprocessing
    """
    
    def __init__(self):
        self.required_columns = ['timestamp', 'usage']
        self.optional_columns = ['temperature', 'humidity', 'appliance_usage']
        self.data_quality_threshold = 0.8  # 80% data quality required
    
    def validate_data(self, df):
        """
        Validate the input data structure and quality
        """
        errors = []
        warnings = []
        
        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check data types
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                errors.append("Invalid timestamp format")
        
        if 'usage' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['usage']):
                try:
                    pd.to_numeric(df['usage'], errors='coerce')
                    warnings.append("Usage column converted to numeric")
                except:
                    errors.append("Usage column cannot be converted to numeric")
        
        # Check for reasonable usage values
        if 'usage' in df.columns:
            usage_values = pd.to_numeric(df['usage'], errors='coerce')
            if usage_values.min() < 0:
                warnings.append("Negative usage values detected")
            if usage_values.max() > 50:  # Unreasonably high for residential
                warnings.append("Unusually high usage values detected (>50 kWh)")
        
        # Check data completeness
        if len(df) > 0:
            completeness = df.notna().sum().sum() / (len(df) * len(df.columns))
            if completeness < self.data_quality_threshold:
                warnings.append(f"Data quality below threshold: {completeness:.2%}")
        
        return errors, warnings
    
    def clean_data(self, df):
        """
        Clean and preprocess the data
        """
        df_clean = df.copy()
        
        # Convert timestamp to datetime
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
            # Remove rows with invalid timestamps
            df_clean = df_clean.dropna(subset=['timestamp'])
        
        # Convert usage to numeric
        if 'usage' in df_clean.columns:
            df_clean['usage'] = pd.to_numeric(df_clean['usage'], errors='coerce')
            # Remove negative values
            df_clean = df_clean[df_clean['usage'] >= 0]
            # Cap extremely high values (potential outliers)
            df_clean['usage'] = df_clean['usage'].clip(upper=50)
        
        # Handle temperature data
        if 'temperature' in df_clean.columns:
            df_clean['temperature'] = pd.to_numeric(df_clean['temperature'], errors='coerce')
            # Fill missing temperature with reasonable defaults
            df_clean['temperature'] = df_clean['temperature'].fillna(22.0)  # Default room temperature
            # Cap extreme temperatures
            df_clean['temperature'] = df_clean['temperature'].clip(lower=-20, upper=50)
        else:
            # Add default temperature if not present
            df_clean['temperature'] = 22.0
        
        # Handle humidity data
        if 'humidity' in df_clean.columns:
            df_clean['humidity'] = pd.to_numeric(df_clean['humidity'], errors='coerce')
            df_clean['humidity'] = df_clean['humidity'].clip(lower=0, upper=100)
            df_clean['humidity'] = df_clean['humidity'].fillna(50.0)  # Default humidity
        else:
            df_clean['humidity'] = 50.0
        
        # Sort by timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
            df_clean = df_clean.reset_index(drop=True)
        
        # Remove duplicates based on timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='first')
        
        return df_clean
    
    def process_uploaded_data(self, df):
        """
        Process data uploaded from CSV files
        """
        # Validate data structure
        errors, warnings = self.validate_data(df)
        
        if errors:
            raise ValueError(f"Data validation errors: {'; '.join(errors)}")
        
        # Clean the data
        df_processed = self.clean_data(df)
        
        # Ensure we have the required columns
        if 'timestamp' not in df_processed.columns:
            # Generate timestamps if missing
            df_processed['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(df_processed)),
                periods=len(df_processed),
                freq='H'
            )
        
        # Handle gaps in time series
        df_processed = self.fill_time_gaps(df_processed)
        
        # Convert timestamps to strings for database storage
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = df_processed['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Add metadata
        df_processed['data_source'] = 'uploaded'
        
        return df_processed
    
    def fill_time_gaps(self, df):
        """
        Fill gaps in time series data
        """
        if len(df) < 2 or 'timestamp' not in df.columns:
            return df
        
        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # Determine the frequency
        time_diff = df.index.to_series().diff().mode()[0]
        
        # Create complete time range
        start_time = df.index.min()
        end_time = df.index.max()
        
        if time_diff <= timedelta(hours=1):
            freq = 'H'
        elif time_diff <= timedelta(days=1):
            freq = 'D'
        else:
            freq = 'H'  # Default to hourly
        
        complete_range = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Reindex to fill gaps
        df_filled = df.reindex(complete_range)
        
        # Forward fill missing values
        df_filled = df_filled.fillna(method='ffill')
        
        # Backward fill any remaining missing values
        df_filled = df_filled.fillna(method='bfill')
        
        # Reset index
        df_filled = df_filled.reset_index()
        df_filled = df_filled.rename(columns={'index': 'timestamp'})
        
        return df_filled
    
    def prepare_for_forecasting(self, df):
        """
        Prepare data specifically for forecasting model
        """
        # Clean the data first
        df_prepared = self.clean_data(df)
        
        # Ensure we have enough data
        if len(df_prepared) < 24:
            # Generate synthetic data if insufficient
            df_prepared = self.generate_synthetic_data(df_prepared)
        
        # Add time-based features
        df_prepared = self.add_time_features(df_prepared)
        
        # Add weather interaction features
        df_prepared = self.add_weather_features(df_prepared)
        
        # Add usage pattern features
        df_prepared = self.add_usage_patterns(df_prepared)
        
        # Handle outliers
        df_prepared = self.handle_outliers(df_prepared)
        
        return df_prepared
    
    def add_time_features(self, df):
        """
        Add time-based features for better forecasting
        """
        if 'timestamp' not in df.columns:
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Add cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Add time-of-day categories
        df['time_category'] = 'other'
        df.loc[(df['hour'] >= 6) & (df['hour'] < 9), 'time_category'] = 'morning'
        df.loc[(df['hour'] >= 9) & (df['hour'] < 17), 'time_category'] = 'day'
        df.loc[(df['hour'] >= 17) & (df['hour'] < 22), 'time_category'] = 'evening'
        df.loc[(df['hour'] >= 22) | (df['hour'] < 6), 'time_category'] = 'night'
        
        return df
    
    def add_weather_features(self, df):
        """
        Add weather-related features
        """
        if 'temperature' not in df.columns:
            return df
        
        df = df.copy()
        
        # Temperature features
        df['temp_squared'] = df['temperature'] ** 2
        df['temp_cubed'] = df['temperature'] ** 3
        
        # Heating/cooling degree days
        df['heating_degree_days'] = np.maximum(0, 18 - df['temperature'])
        df['cooling_degree_days'] = np.maximum(0, df['temperature'] - 24)
        
        # Temperature categories
        df['temp_category'] = 'moderate'
        df.loc[df['temperature'] < 15, 'temp_category'] = 'cold'
        df.loc[df['temperature'] > 25, 'temp_category'] = 'hot'
        
        # Humidity interaction (if available)
        if 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] + 0.5 * df['humidity'] / 100
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        
        return df
    
    def add_usage_patterns(self, df):
        """
        Add usage pattern features
        """
        if 'usage' not in df.columns:
            return df
        
        df = df.copy()
        
        # Rolling statistics
        for window in [6, 12, 24, 48, 168]:  # 6h, 12h, 1d, 2d, 1w
            df[f'usage_rolling_mean_{window}'] = df['usage'].rolling(window=window, min_periods=1).mean()
            df[f'usage_rolling_std_{window}'] = df['usage'].rolling(window=window, min_periods=1).std()
            df[f'usage_rolling_max_{window}'] = df['usage'].rolling(window=window, min_periods=1).max()
            df[f'usage_rolling_min_{window}'] = df['usage'].rolling(window=window, min_periods=1).min()
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
            df[f'usage_lag_{lag}'] = df['usage'].shift(lag)
        
        # Difference features
        df['usage_diff_1'] = df['usage'].diff(1)
        df['usage_diff_24'] = df['usage'].diff(24)
        df['usage_diff_168'] = df['usage'].diff(168)
        
        # Usage rate of change
        df['usage_roc_1'] = df['usage'].pct_change(1)
        df['usage_roc_24'] = df['usage'].pct_change(24)
        
        return df
    
    def handle_outliers(self, df):
        """
        Handle outliers in the data
        """
        if 'usage' not in df.columns:
            return df
        
        df = df.copy()
        
        # Calculate IQR for usage
        Q1 = df['usage'].quantile(0.25)
        Q3 = df['usage'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing them
        df['usage_original'] = df['usage']
        df['usage'] = df['usage'].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        # Mark outliers
        df['is_outlier'] = (
            (df['usage_original'] < lower_bound) | 
            (df['usage_original'] > upper_bound)
        ).astype(int)
        
        return df
    
    def generate_synthetic_data(self, df):
        """
        Generate synthetic data when insufficient historical data is available
        """
        if len(df) == 0:
            # Generate completely synthetic data
            start_time = datetime.now() - timedelta(days=7)
            timestamps = pd.date_range(start=start_time, periods=168, freq='H')
            
            synthetic_data = []
            for ts in timestamps:
                # Create realistic usage pattern
                base_usage = 2.0
                hour_factor = 1 + 0.5 * np.sin(2 * np.pi * ts.hour / 24)
                weekend_factor = 1.2 if ts.dayofweek >= 5 else 1.0
                noise = np.random.normal(0, 0.3)
                
                usage = max(0.5, base_usage * hour_factor * weekend_factor + noise)
                temperature = 20 + 10 * np.sin(2 * np.pi * ts.hour / 24) + np.random.normal(0, 2)
                
                synthetic_data.append({
                    'timestamp': ts,
                    'usage': usage,
                    'temperature': temperature,
                    'humidity': 50 + np.random.normal(0, 10),
                    'data_source': 'synthetic'
                })
            
            return pd.DataFrame(synthetic_data)
        
        else:
            # Extend existing data
            last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])
            avg_usage = df['usage'].mean()
            
            # Generate additional data points
            additional_points = 48 - len(df)  # Ensure at least 48 points
            if additional_points > 0:
                additional_data = []
                for i in range(1, additional_points + 1):
                    new_timestamp = last_timestamp + timedelta(hours=i)
                    
                    # Use pattern from existing data
                    hour_factor = 1 + 0.3 * np.sin(2 * np.pi * new_timestamp.hour / 24)
                    usage = avg_usage * hour_factor + np.random.normal(0, 0.2)
                    
                    additional_data.append({
                        'timestamp': new_timestamp,
                        'usage': max(0.1, usage),
                        'temperature': 22 + np.random.normal(0, 3),
                        'humidity': 50 + np.random.normal(0, 10),
                        'data_source': 'synthetic'
                    })
                
                additional_df = pd.DataFrame(additional_data)
                return pd.concat([df, additional_df], ignore_index=True)
        
        return df
    
    def get_data_summary(self, df):
        """
        Get summary statistics of the processed data
        """
        if len(df) == 0:
            return {"error": "No data to summarize"}
        
        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in df.columns else "N/A",
                "end": df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in df.columns else "N/A"
            },
            "usage_statistics": {
                "mean": df['usage'].mean() if 'usage' in df.columns else 0,
                "median": df['usage'].median() if 'usage' in df.columns else 0,
                "std": df['usage'].std() if 'usage' in df.columns else 0,
                "min": df['usage'].min() if 'usage' in df.columns else 0,
                "max": df['usage'].max() if 'usage' in df.columns else 0
            },
            "data_quality": {
                "completeness": df.notna().sum().sum() / (len(df) * len(df.columns)),
                "missing_values": df.isna().sum().sum(),
                "duplicate_records": df.duplicated().sum()
            }
        }
        
        return summary
    
    def export_processed_data(self, df, filepath):
        """
        Export processed data to CSV
        """
        try:
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False