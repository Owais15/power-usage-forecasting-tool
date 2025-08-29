import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import warnings
from typing import Optional, Any
warnings.filterwarnings('ignore')

class PowerForecastingModel:
    """
    Power Usage Forecasting Model for predicting electricity consumption
    Uses multiple regression techniques for accurate forecasting
    """
    
    def __init__(self, model_path='data/trained_model.pkl'):
        self.model_path = model_path
        self.model = None  # type: Optional[Any]
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
        # Model parameters
        self.forecast_horizon = 24  # Hours to predict
        self.lookback_window = 168  # Hours to look back (1 week)
        
        # Load existing model if available
        self.load_model()
    
    def extract_features(self, data):
        """
        Extract relevant features from power usage data
        """
        df = data.copy()
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Weather features (if available)
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_lag1'] = df['temperature'].shift(1)
        else:
            # Use default temperature if not available
            df['temperature'] = 22.0  # Default temperature
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_lag1'] = df['temperature']
        
        # Lag features for usage
        if 'usage' in df.columns:
            df['usage_lag1'] = df['usage'].shift(1)
            df['usage_lag2'] = df['usage'].shift(2)
            df['usage_lag24'] = df['usage'].shift(24)  # Same hour previous day
            df['usage_lag168'] = df['usage'].shift(168)  # Same hour previous week
            
            # Rolling statistics
            df['usage_rolling_mean_24'] = df['usage'].rolling(window=24, min_periods=1).mean()
            df['usage_rolling_std_24'] = df['usage'].rolling(window=24, min_periods=1).std()
            df['usage_rolling_mean_168'] = df['usage'].rolling(window=168, min_periods=1).mean()
        
        # Peak hours indicator
        df['is_peak_hour'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        df['is_off_peak'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Season indicator
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] >= 12) | (df['month'] <= 2)).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def prepare_training_data(self, data):
        """
        Prepare data for training the forecasting model
        """
        df = self.extract_features(data)
        
        # Define feature columns (excluding target variable)
        feature_columns = [col for col in df.columns if col != 'usage']
        self.feature_columns = feature_columns
        
        # Prepare X and y
        X = df[feature_columns].values
        y = df['usage'].values
        
        # Remove rows with NaN values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Check if we have enough valid data
        if len(X) < 10:  # Need at least 10 data points
            raise ValueError(f"Insufficient valid data for training. Only {len(X)} valid data points available.")
        
        return X, y
    
    def train_model(self, data):
        """
        Train the forecasting model using historical data
        """
        print("Training power usage forecasting model...")
        
        try:
            # Prepare training data
            X, y = self.prepare_training_data(data)
            
            if len(X) < 50:
                print("Warning: Insufficient data for training. Need at least 50 data points.")
                return False
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Try different models and choose the best one
            models = {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            best_model = None
            best_score = float('inf')
            best_model_name = None
            
            for name, model in models.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate error
                mse = mean_squared_error(y_test, y_pred)
                
                print(f"{name} - MSE: {mse:.4f}")
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            
            self.model = best_model
            self.is_trained = True
            
            # Calculate and display metrics
            assert self.model is not None, "Model should be set after training"
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nBest Model: {best_model_name}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"Mean Squared Error: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Save the trained model
            self.save_model()
            
            return True
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            return False
    
    def predict(self, data, hours_ahead=None):
        """
        Generate power usage forecast
        """
        if not self.is_trained or self.model is None:
            print("Model not trained. Training with provided data...")
            if not self.train_model(data):
                return self._generate_dummy_forecast(hours_ahead or self.forecast_horizon)
        
        if hours_ahead is None:
            hours_ahead = self.forecast_horizon
        
        try:
            # Check if model is available
            if self.model is None:
                return self._generate_dummy_forecast(hours_ahead)
            
            # Extract features from input data
            df = self.extract_features(data)
            
            # Get the last available data point
            last_timestamp = df.index[-1]
            last_usage = df['usage'].iloc[-1]
            
            # Generate future timestamps
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=hours_ahead,
                freq='H'
            )
            
            predictions = []
            confidence_upper = []
            confidence_lower = []
            
            # Use the last few data points for prediction
            recent_data = df.tail(24).copy()
            
            for i, future_time in enumerate(future_timestamps):
                # Create feature vector for future time
                future_features = self._create_future_features(
                    future_time, recent_data, last_usage
                )
                
                # Scale features
                future_features_scaled = self.scaler.transform([future_features])
                
                # Make prediction (with explicit None check)
                assert self.model is not None, "Model should not be None at this point"
                prediction = self.model.predict(future_features_scaled)[0]
                
                # Add some realistic constraints
                prediction = max(0.1, min(prediction, 10.0))  # Reasonable bounds
                
                predictions.append(prediction)
                
                # Calculate confidence intervals (simplified)
                std_dev = np.std(data['usage']) if 'usage' in data.columns else 0.5
                confidence_upper.append(prediction + 1.96 * std_dev)
                confidence_lower.append(max(0, prediction - 1.96 * std_dev))
                
                # Update last_usage for next iteration
                last_usage = prediction
            
            return {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in future_timestamps],
                'predictions': predictions,
                'confidence_upper': confidence_upper,
                'confidence_lower': confidence_lower
            }
        
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return self._generate_dummy_forecast(hours_ahead)
    
    def _create_future_features(self, timestamp, recent_data, last_usage):
        """
        Create feature vector for future timestamp
        """
        # Time-based features
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        month = timestamp.month
        is_weekend = int(timestamp.dayofweek >= 5)
        
        # Cyclical encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # Weather features (use recent average or default)
        if 'temperature' in recent_data.columns:
            temperature = recent_data['temperature'].mean()
        else:
            temperature = 22.0  # Default temperature
        
        temp_squared = temperature ** 2
        temp_lag1 = temperature
        
        # Usage lag features
        usage_lag1 = last_usage
        usage_lag2 = recent_data['usage'].iloc[-2] if len(recent_data) >= 2 else last_usage
        usage_lag24 = recent_data['usage'].iloc[-24] if len(recent_data) >= 24 else last_usage
        usage_lag168 = last_usage  # Simplified for demo
        
        # Rolling statistics
        usage_rolling_mean_24 = recent_data['usage'].mean()
        usage_rolling_std_24 = recent_data['usage'].std()
        usage_rolling_mean_168 = recent_data['usage'].mean()
        
        # Peak hours indicator
        is_peak_hour = int(18 <= hour <= 22)
        is_off_peak = int(hour >= 22 or hour <= 6)
        
        # Season indicator
        is_summer = int(6 <= month <= 8)
        is_winter = int(month >= 12 or month <= 2)
        
        # Combine all features (order must match training data)
        features = [
            hour, day_of_week, month, is_weekend,
            hour_sin, hour_cos, day_sin, day_cos,
            temperature, temp_squared, temp_lag1,
            usage_lag1, usage_lag2, usage_lag24, usage_lag168,
            usage_rolling_mean_24, usage_rolling_std_24, usage_rolling_mean_168,
            is_peak_hour, is_off_peak, is_summer, is_winter
        ]
        
        return np.array(features)
    
    def _generate_dummy_forecast(self, hours_ahead):
        """
        Generate dummy forecast when model is not available
        """
        print("Generating dummy forecast...")
        
        # Create realistic-looking dummy data
        base_usage = 2.5
        future_timestamps = pd.date_range(
            start=datetime.now() + timedelta(hours=1),
            periods=hours_ahead,
            freq='H'
        )
        
        predictions = []
        for ts in future_timestamps:
            # Create pattern based on hour of day
            hour_factor = 1 + 0.3 * np.sin(2 * np.pi * ts.hour / 24)
            daily_pattern = base_usage * hour_factor
            
            # Add some random variation
            noise = np.random.normal(0, 0.2)
            prediction = max(0.5, daily_pattern + noise)
            predictions.append(prediction)
        
        return {
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in future_timestamps],
            'predictions': predictions,
            'confidence_upper': [p + 0.5 for p in predictions],
            'confidence_lower': [max(0, p - 0.5) for p in predictions]
        }
    
    def save_model(self):
        """
        Save the trained model to disk
        """
        if self.model is not None:
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_columns': self.feature_columns,
                    'is_trained': self.is_trained
                }
                
                with open(self.model_path, 'wb') as f:
                    pickle.dump(model_data, f)
                
                print(f"Model saved to {self.model_path}")
            except Exception as e:
                print(f"Error saving model: {str(e)}")
    
    def load_model(self):
        """
        Load a previously trained model from disk
        """
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_columns = model_data['feature_columns']
                self.is_trained = model_data['is_trained']
                
                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.model = None
                self.is_trained = False
    
    def get_model_info(self):
        """
        Get information about the current model
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "status": "Model loaded",
            "model_type": type(self.model).__name__,
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_columns) if self.feature_columns else 0,
            "forecast_horizon": self.forecast_horizon
        }