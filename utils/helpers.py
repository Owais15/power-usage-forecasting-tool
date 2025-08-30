import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

def allowed_file(filename):
    """
    Check if uploaded file has allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def generate_sample_data(user_id=None):
    """
    Generate sample power usage data for demonstration
    """
    from utils.database import insert_power_usage
    
    # Generate 7 days of hourly data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    # Create time range
    time_range = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # Generate realistic usage patterns
    base_usage = 2.5  # Base usage in kWh
    usage_data = []
    
    for timestamp in time_range:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        is_weekend = day_of_week >= 5
        
        # Base usage with daily patterns
        usage = base_usage
        
        # Add time-of-day variation
        if 6 <= hour <= 8:  # Morning peak
            usage += 1.5
        elif 18 <= hour <= 22:  # Evening peak
            usage += 2.0
        elif 22 <= hour or hour <= 6:  # Night (low usage)
            usage -= 1.0
        
        # Add weekend variation
        if is_weekend:
            usage += 0.5
        
        # Add some randomness
        usage += np.random.normal(0, 0.3)
        
        # Ensure usage is positive
        usage = max(0.1, usage)
        
        # Generate temperature (correlated with usage)
        base_temp = 22.0
        if 6 <= hour <= 8:  # Morning
            temp = base_temp + 2 + np.random.normal(0, 1)
        elif 18 <= hour <= 22:  # Evening
            temp = base_temp + 3 + np.random.normal(0, 1)
        else:
            temp = base_temp + np.random.normal(0, 2)
        
        # Generate humidity
        humidity = 50 + np.random.normal(0, 10)
        humidity = max(20, min(80, humidity))
        
        # Insert into database
        insert_power_usage(
            timestamp=timestamp,
            usage=round(usage, 2),
            temperature=round(temp, 1),
            humidity=round(humidity, 1),
            appliance_usage='sample_data',
            user_id=user_id
        )
        
        usage_data.append({
            'timestamp': timestamp,
            'usage': usage,
            'temperature': temp,
            'humidity': humidity
        })
    
    return pd.DataFrame(usage_data)

def format_usage(usage_kwh):
    """
    Format usage value for display
    """
    if usage_kwh < 1:
        return f"{usage_kwh * 1000:.0f} W"
    else:
        return f"{usage_kwh:.2f} kWh"

def format_cost(cost):
    """
    Format cost value for display
    """
    return f"${cost:.2f}"

def format_percentage(value):
    """
    Format percentage value for display
    """
    return f"{value:.1%}"

def get_time_period_display(hours):
    """
    Get human-readable time period
    """
    if hours < 24:
        return f"{hours} hours"
    elif hours < 168:
        days = hours // 24
        return f"{days} days"
    else:
        weeks = hours // 168
        return f"{weeks} weeks"

def calculate_savings_percentage(current_cost, optimized_cost):
    """
    Calculate percentage savings
    """
    if current_cost <= 0:
        return 0
    return ((current_cost - optimized_cost) / current_cost) * 100

def get_impact_color(impact):
    """
    Get color for impact level
    """
    colors = {
        'high': '#e74c3c',    # Red
        'medium': '#f39c12',  # Orange
        'low': '#27ae60'      # Green
    }
    return colors.get(impact, '#95a5a6')  # Default gray

def validate_date_range(start_date, end_date):
    """
    Validate date range input
    """
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        if start >= end:
            return False, "Start date must be before end date"
        
        if end > datetime.now():
            return False, "End date cannot be in the future"
        
        if start < datetime.now() - timedelta(days=365):
            return False, "Start date cannot be more than 1 year ago"
        
        return True, None
    except:
        return False, "Invalid date format"

def create_export_filename(prefix="power_usage", extension="csv"):
    """
    Create a unique filename for exports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def sanitize_filename(filename):
    """
    Sanitize filename for safe file operations
    """
    # Remove or replace unsafe characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100-len(ext)] + ext
    
    return filename

def get_file_size_mb(filepath):
    """
    Get file size in MB
    """
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        return round(size_bytes / (1024 * 1024), 2)
    return 0

def is_valid_csv(filepath):
    """
    Check if file is a valid CSV
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'windows-1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, nrows=5, encoding=encoding)
                required_cols = ['timestamp', 'usage']
                if all(col in df.columns for col in required_cols):
                    return True
            except:
                continue
        return False
    except:
        return False

def detect_file_encoding(filepath):
    """
    Detect the encoding of a file using common encodings
    """
    # Check if file is a text file (not binary like Excel)
    if not os.path.exists(filepath):
        return 'utf-8'
    
    # Check file extension to avoid processing binary files
    file_extension = filepath.lower().split('.')[-1] if '.' in filepath else ''
    if file_extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
        print(f"Warning: detect_file_encoding called on Excel file: {filepath}")
        return 'utf-8'  # Default for Excel files
    
    encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                f.read()
            return encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error reading file {filepath} with encoding {encoding}: {str(e)}")
            continue
    
    return 'utf-8'  # Default fallback

def get_weather_icon(temperature):
    """
    Get weather icon based on temperature
    """
    if temperature < 0:
        return "❄️"  # Snow
    elif temperature < 15:
        return "🌤️"  # Partly cloudy
    elif temperature < 25:
        return "☀️"  # Sunny
    else:
        return "🔥"  # Hot

def get_usage_status(usage, threshold=4.0):
    """
    Get usage status based on threshold
    """
    if usage > threshold * 1.5:
        return "critical", "Very High Usage"
    elif usage > threshold:
        return "warning", "High Usage"
    elif usage > threshold * 0.5:
        return "normal", "Normal Usage"
    else:
        return "good", "Low Usage"

def format_timestamp(timestamp):
    """
    Format timestamp for display
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 0:
        return timestamp.strftime("%b %d, %Y")
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"

def calculate_carbon_footprint(usage_kwh, emission_factor=0.5):
    """
    Calculate carbon footprint from electricity usage
    Emission factor in kg CO2 per kWh (varies by region)
    """
    return usage_kwh * emission_factor

def get_efficiency_rating(usage_kwh, household_size=1):
    """
    Get efficiency rating based on usage per person
    """
    usage_per_person = usage_kwh / household_size
    
    if usage_per_person < 2:
        return "A+", "Excellent"
    elif usage_per_person < 4:
        return "A", "Good"
    elif usage_per_person < 6:
        return "B", "Average"
    elif usage_per_person < 8:
        return "C", "Below Average"
    else:
        return "D", "Poor"

def create_summary_stats(data):
    """
    Create summary statistics from usage data
    """
    if len(data) == 0:
        return {}
    
    stats = {
        'total_usage': data['usage'].sum(),
        'average_usage': data['usage'].mean(),
        'max_usage': data['usage'].max(),
        'min_usage': data['usage'].min(),
        'usage_std': data['usage'].std(),
        'total_hours': len(data),
        'peak_hour': data.loc[data['usage'].idxmax(), 'timestamp'] if 'timestamp' in data.columns else None
    }
    
    if 'temperature' in data.columns:
        stats['avg_temperature'] = data['temperature'].mean()
        stats['temp_correlation'] = data['usage'].corr(data['temperature'])
    
    return stats

def generate_report_data(start_date, end_date):
    """
    Generate comprehensive report data
    """
    from utils.database import get_usage_by_date_range, get_usage_statistics
    
    # Get usage data
    usage_data = get_usage_by_date_range(start_date, end_date)
    
    if len(usage_data) == 0:
        return None
    
    # Calculate statistics
    stats = create_summary_stats(usage_data)
    
    # Calculate costs
    from models.optimization import OptimizationEngine
    opt_engine = OptimizationEngine()
    current_cost = opt_engine.calculate_cost(usage_data)
    optimized_cost = opt_engine.calculate_optimized_cost(usage_data)
    
    # Get optimization suggestions
    suggestions = opt_engine.get_optimization_suggestions(usage_data)
    
    report_data = {
        'period': {
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': (end_date - start_date).days
        },
        'usage_stats': stats,
        'cost_analysis': {
            'current_cost': current_cost,
            'optimized_cost': optimized_cost,
            'potential_savings': current_cost - optimized_cost,
            'savings_percentage': calculate_savings_percentage(current_cost, optimized_cost)
        },
        'optimization_suggestions': suggestions,
        'data_points': len(usage_data)
    }
    
    return report_data 