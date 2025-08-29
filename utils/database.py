import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from config import Config

def get_db_connection():
    """
    Get a connection to the SQLite database
    """
    # Ensure database directory exists
    os.makedirs(os.path.dirname(Config.DATABASE_PATH), exist_ok=True)
    
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database with required tables
    """
    conn = get_db_connection()
    
    # Create users table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
    ''')
    
    # Create power_usage table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS power_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            usage REAL NOT NULL,
            temperature REAL,
            humidity REAL,
            appliance_usage TEXT,
            data_source TEXT DEFAULT 'manual',
            user_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create settings table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            user_id INTEGER,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create forecasts table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            predicted_usage REAL NOT NULL,
            confidence_upper REAL,
            confidence_lower REAL,
            model_version TEXT,
            user_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create optimization_suggestions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS optimization_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suggestion_type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            impact TEXT,
            potential_savings REAL,
            actions TEXT,
            user_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert default settings
    default_settings = [
        ('electricity_rate', '0.12'),
        ('peak_hours', '18:00-22:00'),
        ('off_peak_hours', '22:00-06:00'),
        ('high_usage_threshold', '4.0'),
        ('enable_notifications', 'false'),
        ('data_retention_days', '365')
    ]
    
    for key, value in default_settings:
        conn.execute('''
            INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)
        ''', (key, value))
    
    conn.commit()
    conn.close()

def insert_power_usage(timestamp, usage, temperature=None, humidity=None, appliance_usage=None, user_id=None):
    """
    Insert a new power usage record
    """
    # Convert timestamp to string if it's a pandas Timestamp or datetime object
    if hasattr(timestamp, 'strftime'):
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    conn = get_db_connection()
    try:
        conn.execute('''
            INSERT INTO power_usage (timestamp, usage, temperature, humidity, appliance_usage, data_source, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, usage, temperature, humidity, appliance_usage, 'manual', user_id))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error inserting power usage: {str(e)}")
        return False
    finally:
        conn.close()

def get_recent_usage(hours=168, user_id=None):
    """
    Get recent power usage data
    """
    conn = get_db_connection()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    if user_id:
        df = pd.read_sql_query('''
            SELECT * FROM power_usage 
            WHERE timestamp >= ? AND user_id = ?
            ORDER BY timestamp DESC
        ''', conn, params=[cutoff_time, user_id])
    else:
        df = pd.read_sql_query('''
            SELECT * FROM power_usage 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        ''', conn, params=[cutoff_time])
    
    # Convert timestamp strings back to datetime
    if len(df) > 0 and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    conn.close()
    return df

def get_usage_by_date_range(start_date, end_date):
    """
    Get power usage data for a specific date range
    """
    conn = get_db_connection()
    
    df = pd.read_sql_query('''
        SELECT * FROM power_usage 
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp
    ''', conn, params=[start_date, end_date])
    
    # Convert timestamp strings back to datetime
    if len(df) > 0 and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    conn.close()
    return df

def get_current_usage():
    """
    Get the most recent power usage
    """
    conn = get_db_connection()
    result = conn.execute('''
        SELECT * FROM power_usage 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''').fetchone()
    conn.close()
    
    return result

def get_usage_statistics(hours=168):
    """
    Get usage statistics for the specified time period
    """
    conn = get_db_connection()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    result = conn.execute('''
        SELECT 
            COUNT(*) as count,
            AVG(usage) as avg_usage,
            MAX(usage) as max_usage,
            MIN(usage) as min_usage,
            SUM(usage) as total_usage,
            AVG(temperature) as avg_temperature
        FROM power_usage 
        WHERE timestamp >= ?
    ''', (cutoff_time,)).fetchone()
    
    conn.close()
    return dict(result)

def save_forecast(forecast_data):
    """
    Save forecast data to database
    """
    conn = get_db_connection()
    
    for timestamp, prediction, upper, lower in zip(
        forecast_data['timestamps'],
        forecast_data['predictions'],
        forecast_data.get('confidence_upper', []),
        forecast_data.get('confidence_lower', [])
    ):
        # Convert timestamp to string if it's a pandas Timestamp or datetime object
        if hasattr(timestamp, 'strftime'):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        conn.execute('''
            INSERT INTO forecasts (timestamp, predicted_usage, confidence_upper, confidence_lower, model_version)
            VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, prediction, upper, lower, 'v1.0'))
    
    conn.commit()
    conn.close()

def get_recent_forecasts(hours=24):
    """
    Get recent forecast data
    """
    conn = get_db_connection()
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    df = pd.read_sql_query('''
        SELECT * FROM forecasts 
        WHERE timestamp >= ? 
        ORDER BY timestamp
    ''', conn, params=[cutoff_time])
    
    # Convert timestamp strings back to datetime
    if len(df) > 0 and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    conn.close()
    return df

def save_optimization_suggestion(suggestion):
    """
    Save optimization suggestion to database
    """
    conn = get_db_connection()
    
    conn.execute('''
        INSERT INTO optimization_suggestions 
        (suggestion_type, title, description, impact, potential_savings, actions)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        suggestion['type'],
        suggestion['title'],
        suggestion['description'],
        suggestion['impact'],
        suggestion.get('potential_savings', 0),
        '; '.join(suggestion.get('actions', []))
    ))
    
    conn.commit()
    conn.close()

def get_recent_suggestions(limit=10):
    """
    Get recent optimization suggestions
    """
    conn = get_db_connection()
    
    df = pd.read_sql_query('''
        SELECT * FROM optimization_suggestions 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', conn, params=[limit])
    
    conn.close()
    return df

def get_setting(key, default=None):
    """
    Get a setting value from the database
    """
    conn = get_db_connection()
    result = conn.execute('''
        SELECT value FROM settings WHERE key = ?
    ''', (key,)).fetchone()
    conn.close()
    
    return result['value'] if result else default

def update_setting(key, value):
    """
    Update a setting value in the database
    """
    conn = get_db_connection()
    conn.execute('''
        INSERT OR REPLACE INTO settings (key, value, updated_at)
        VALUES (?, ?, ?)
    ''', (key, value, datetime.now()))
    conn.commit()
    conn.close()

def get_all_settings():
    """
    Get all settings from the database
    """
    conn = get_db_connection()
    result = conn.execute('SELECT key, value FROM settings').fetchall()
    conn.close()
    
    return {row['key']: row['value'] for row in result}

def cleanup_old_data(retention_days=365):
    """
    Clean up old data based on retention policy
    """
    conn = get_db_connection()
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    # Delete old power usage data
    conn.execute('''
        DELETE FROM power_usage 
        WHERE timestamp < ?
    ''', (cutoff_date,))
    
    # Delete old forecasts
    conn.execute('''
        DELETE FROM forecasts 
        WHERE timestamp < ?
    ''', (cutoff_date,))
    
    # Delete old optimization suggestions (keep last 30 days)
    suggestion_cutoff = datetime.now() - timedelta(days=30)
    conn.execute('''
        DELETE FROM optimization_suggestions 
        WHERE created_at < ?
    ''', (suggestion_cutoff,))
    
    conn.commit()
    conn.close()

def export_data_to_csv(start_date=None, end_date=None, filepath='export.csv'):
    """
    Export power usage data to CSV
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    df = get_usage_by_date_range(start_date, end_date)
    df.to_csv(filepath, index=False)
    return filepath

def get_database_info():
    """
    Get database statistics and information
    """
    conn = get_db_connection()
    
    # Get table sizes
    tables = ['power_usage', 'forecasts', 'optimization_suggestions', 'settings']
    table_info = {}
    
    for table in tables:
        result = conn.execute(f'SELECT COUNT(*) as count FROM {table}').fetchone()
        table_info[table] = result['count']
    
    # Get database size
    db_size = os.path.getsize(Config.DATABASE_PATH) if os.path.exists(Config.DATABASE_PATH) else 0
    
    # Get oldest and newest records
    oldest = conn.execute('SELECT MIN(timestamp) as oldest FROM power_usage').fetchone()
    newest = conn.execute('SELECT MAX(timestamp) as newest FROM power_usage').fetchone()
    
    conn.close()
    
    return {
        'table_counts': table_info,
        'database_size_mb': round(db_size / (1024 * 1024), 2),
        'oldest_record': oldest['oldest'] if oldest['oldest'] else None,
        'newest_record': newest['newest'] if newest['newest'] else None
    } 