import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Basic Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'power-usage-forecasting-secret-key-2025'
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or 'database/power_usage.db'
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Machine Learning Model Configuration
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'data/trained_model.pkl'
    FORECAST_HORIZON = int(os.environ.get('FORECAST_HORIZON', 24))  # Hours to forecast
    MODEL_UPDATE_INTERVAL = int(os.environ.get('MODEL_UPDATE_INTERVAL', 24))  # Hours between model updates
    
    # Data Processing Configuration
    DATA_RETENTION_DAYS = int(os.environ.get('DATA_RETENTION_DAYS', 365))  # Days to keep historical data
    MINIMUM_DATA_POINTS = int(os.environ.get('MINIMUM_DATA_POINTS', 48))  # Minimum data points for forecasting
    
    # Optimization Configuration
    DEFAULT_ELECTRICITY_RATE = float(os.environ.get('DEFAULT_ELECTRICITY_RATE', 0.12))  # $/kWh
    PEAK_HOURS = os.environ.get('PEAK_HOURS', '18:00-22:00').split(',')
    OFF_PEAK_HOURS = os.environ.get('OFF_PEAK_HOURS', '22:00-06:00').split(',')
    
    # Green Energy Configuration
    SOLAR_PRODUCTION_HOURS = os.environ.get('SOLAR_PRODUCTION_HOURS', '06:00-18:00')
    GREEN_ENERGY_RATE = float(os.environ.get('GREEN_ENERGY_RATE', 0.08))  # $/kWh for green energy
    
    # API Configuration
    API_RATE_LIMIT = os.environ.get('API_RATE_LIMIT', '100 per hour')
    API_TIMEOUT = int(os.environ.get('API_TIMEOUT', 30))  # seconds
    
    # Notification Configuration
    ENABLE_NOTIFICATIONS = os.environ.get('ENABLE_NOTIFICATIONS', 'False').lower() == 'true'
    HIGH_USAGE_THRESHOLD = float(os.environ.get('HIGH_USAGE_THRESHOLD', 5.0))  # kWh
    
    # Chart and Visualization Configuration
    CHART_COLORS = {
        'primary': '#3498db',
        'secondary': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'success': '#27ae60',
        'info': '#17a2b8'
    }
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'logs/app.log')
    
    # Feature Flags
    ENABLE_WEATHER_INTEGRATION = os.environ.get('ENABLE_WEATHER_INTEGRATION', 'True').lower() == 'true'
    ENABLE_REAL_TIME_UPDATES = os.environ.get('ENABLE_REAL_TIME_UPDATES', 'True').lower() == 'true'
    ENABLE_EXPORT_FUNCTIONALITY = os.environ.get('ENABLE_EXPORT_FUNCTIONALITY', 'True').lower() == 'true'
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create necessary directories
        os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
        
        # Set up logging
        if Config.LOG_FILE:
            os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # Use in-memory database for development
    DATABASE_PATH = 'database/dev_power_usage.db'
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Enable all features in development
    ENABLE_WEATHER_INTEGRATION = True
    ENABLE_REAL_TIME_UPDATES = True
    ENABLE_EXPORT_FUNCTIONALITY = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use temporary database for testing
    DATABASE_PATH = ':memory:'
    
    # Disable CSRF protection for testing
    WTF_CSRF_ENABLED = False
    
    # Shorter forecast horizon for faster tests
    FORECAST_HORIZON = 6
    MINIMUM_DATA_POINTS = 12

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use environment variables for sensitive data in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Production database path
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or '/app/database/power_usage.db'
    
    # Secure session cookies in production
    SESSION_COOKIE_SECURE = True
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    LOG_FILE = '/app/logs/production.log'
    
    # Rate limiting in production
    API_RATE_LIMIT = '50 per hour'
    
    @staticmethod
    def init_app(app):
        """Initialize application with production configuration"""
        # Check for required environment variables
        if not ProductionConfig.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable must be set in production")
        
        # Call parent init_app
        Config.init_app(app)

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])