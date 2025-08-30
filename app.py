from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from werkzeug.utils import secure_filename
from models.forecasting_model import PowerForecastingModel
from models.data_processing import DataProcessor
from models.optimization import OptimizationEngine
from utils.database import init_db, get_db_connection
from utils.helpers import allowed_file, generate_sample_data, detect_file_encoding
from utils.auth import login_required, authenticate_user, create_user, logout_user, get_current_user, hash_password, verify_password
from config import Config
import math

app = Flask(__name__)
app.config.from_object(Config)

# Initialize configuration (creates necessary directories)
Config.init_app(app)

# Initialize components
forecasting_model = PowerForecastingModel()
data_processor = DataProcessor()
optimization_engine = OptimizationEngine()

# Initialize database
init_db()

def sanitize_for_json(obj):
    import pandas as pd
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif obj is None or (isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))):
        return 0
    elif hasattr(obj, 'item'):
        # Handles numpy types
        return obj.item()
    return obj

@app.route('/')
def index():
    """Home page with basic information about the tool"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return render_template('login.html')
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            
            # Update last login time
            conn = get_db_connection()
            conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(), user['id']))
            conn.commit()
            conn.close()
            
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            else:
                flash('Login successful! Welcome back.', 'success')
                return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User registration page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([username, email, password, confirm_password]):
            flash('Please fill in all fields.', 'error')
            return render_template('signup.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('signup.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'error')
            return render_template('signup.html')
        
        # Create user
        success, message = create_user(username, email, password)
        if success:
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    try:
        user = get_current_user()
        
        # Get user statistics
        conn = get_db_connection()
        
        # Get total records count
        total_records = conn.execute(
            'SELECT COUNT(*) as count FROM power_usage WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()['count']
        
        # Get total usage
        total_usage_result = conn.execute(
            'SELECT SUM(usage) as total FROM power_usage WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()
        total_usage = float(total_usage_result['total']) if total_usage_result['total'] else 0.0
        
        # Get average usage
        avg_usage_result = conn.execute(
            'SELECT AVG(usage) as avg_usage FROM power_usage WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()
        avg_usage = float(avg_usage_result['avg_usage']) if avg_usage_result['avg_usage'] else 0.0
        
        # Get first and last data entry
        first_entry = conn.execute(
            'SELECT timestamp FROM power_usage WHERE user_id = ? ORDER BY timestamp ASC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
        
        last_entry = conn.execute(
            'SELECT timestamp FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1',
            (session['user_id'],)
        ).fetchone()
        
        # Get settings count
        settings_count = conn.execute(
            'SELECT COUNT(*) as count FROM settings WHERE user_id = ?',
            (session['user_id'],)
        ).fetchone()['count']
        
        conn.close()
        
        # Calculate days since first entry
        days_active = 0
        if first_entry and first_entry['timestamp']:
            from datetime import datetime
            first_date = datetime.fromisoformat(first_entry['timestamp'].replace('Z', '+00:00'))
            days_active = (datetime.now() - first_date).days
        
        user_stats = {
            'total_records': total_records,
            'total_usage': total_usage,
            'avg_usage': avg_usage,
            'days_active': days_active,
            'settings_count': settings_count,
            'first_entry': first_entry['timestamp'] if first_entry else None,
            'last_entry': last_entry['timestamp'] if last_entry else None
        }
        
        return render_template('profile.html', user=user, stats=user_stats)
        
    except Exception as e:
        flash(f'Error loading profile: {str(e)}', 'error')
        return render_template('profile.html', user=user, stats={})

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard showing current usage and basic statistics"""
    try:
        conn = get_db_connection()
        
        # Get recent power usage data from the last 7 days
        from datetime import datetime, timedelta
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        # Since @login_required ensures user_id is in session, use it directly
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE timestamp >= ? AND user_id = ? ORDER BY timestamp DESC", 
            conn, params=[seven_days_ago, session['user_id']]
        )
        
        print(f"Dashboard: Querying data from {seven_days_ago} onwards for user {session['user_id']}")
        print(f"Dashboard: Found {len(recent_data)} records in database")
        
        # Convert timestamp strings back to datetime for processing
        if len(recent_data) > 0 and 'timestamp' in recent_data.columns:
            recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
        
        # Get current statistics
        if len(recent_data) > 0:
            print(f"Dashboard: Latest timestamp: {recent_data.iloc[0]['timestamp']}")
            print(f"Dashboard: Earliest timestamp: {recent_data.iloc[-1]['timestamp']}")
            
            # Check data sources
            if 'data_source' in recent_data.columns:
                source_counts = recent_data['data_source'].value_counts()
                print(f"Dashboard: Data sources: {source_counts.to_dict()}")
            
            current_usage = float(recent_data.iloc[0]['usage'])
            avg_usage = float(recent_data['usage'].mean())
            total_usage = float(recent_data['usage'].sum())
            
            # Prepare data for charts (show more data to include uploaded data)
            # Since data is ordered DESC, we want the first 100 records (most recent)
            chart_data = {
                'timestamps': recent_data['timestamp'].head(100).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'usage': [float(x) for x in recent_data['usage'].head(100).tolist()],
                'temperature': [float(x) for x in recent_data['temperature'].head(100).tolist()]
            }
        else:
            # New user or no data - start with zeros
            print(f"Dashboard: No data found for user {session['user_id']}")
            current_usage = 0.0
            avg_usage = 0.0
            total_usage = 0.0
            chart_data = {
                'timestamps': [],
                'usage': [],
                'temperature': []
            }
        
        conn.close()
        
        return render_template('dashboard.html', 
                             current_usage=current_usage,
                             avg_usage=avg_usage,
                             total_usage=total_usage,
                             chart_data=chart_data)
    
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('dashboard.html', 
                             current_usage=0,
                             avg_usage=0,
                             total_usage=0,
                             chart_data={'timestamps': [], 'usage': [], 'temperature': []})

@app.route('/forecast')
@login_required
def forecast():
    """Forecast page showing predicted power usage"""
    try:
        conn = get_db_connection()
        
        # Get historical data for forecasting
        # Since @login_required ensures user_id is in session, use it directly
        historical_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 168", 
            conn, params=[session['user_id']]
        )
        
        # Convert timestamp strings back to datetime for processing
        if len(historical_data) > 0 and 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        if len(historical_data) > 0:
            try:
                # Prepare data for forecasting
                processed_data = data_processor.prepare_for_forecasting(historical_data)
                
                # Generate forecast
                forecast_data = forecasting_model.predict(processed_data)
                
                # Validate forecast data structure
                if not isinstance(forecast_data, dict) or 'predictions' not in forecast_data:
                    raise ValueError("Invalid forecast data structure")
                
                # Prepare data for visualization
                forecast_chart_data = {
                    'historical_timestamps': historical_data['timestamp'].tail(48).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'historical_usage': [float(x) for x in historical_data['usage'].tail(48).tolist()],
                    'forecast_timestamps': forecast_data.get('timestamps', []),
                    'forecast_usage': [float(x) for x in forecast_data.get('predictions', [])],
                    'confidence_upper': [float(x) for x in forecast_data.get('confidence_upper', [])],
                    'confidence_lower': [float(x) for x in forecast_data.get('confidence_lower', [])]
                }
                
                # Calculate forecast summary with safety checks
                predictions = forecast_data.get('predictions', [])
                timestamps = forecast_data.get('timestamps', [])
                
                if len(predictions) >= 24:
                    forecast_summary = {
                        'next_24h_avg': float(np.mean(predictions[:24])),
                        'peak_hour': timestamps[np.argmax(predictions[:24])] if len(timestamps) > 0 else 'N/A',
                        'peak_usage': float(max(predictions[:24])),
                        'total_predicted': float(sum(predictions[:24]))
                    }
                else:
                    # Fallback for insufficient predictions
                    forecast_summary = {
                        'next_24h_avg': 0.0,
                        'peak_hour': 'N/A',
                        'peak_usage': 0.0,
                        'total_predicted': 0.0
                    }
            except Exception as forecast_error:
                print(f"Forecast generation error: {str(forecast_error)}")
                # Fallback to dummy data
                forecast_chart_data = {
                    'historical_timestamps': historical_data['timestamp'].tail(48).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'historical_usage': [float(x) for x in historical_data['usage'].tail(48).tolist()],
                    'forecast_timestamps': [],
                    'forecast_usage': [],
                    'confidence_upper': [],
                    'confidence_lower': []
                }
                forecast_summary = {
                    'next_24h_avg': 0.0,
                    'peak_hour': 'N/A',
                    'peak_usage': 0.0,
                    'total_predicted': 0.0
                }
        else:
            # New user or no data - start with zeros
            forecast_chart_data = {
                'historical_timestamps': [],
                'historical_usage': [],
                'forecast_timestamps': [],
                'forecast_usage': [],
                'confidence_upper': [],
                'confidence_lower': []
            }
            forecast_summary = {
                'next_24h_avg': 0.0,
                'peak_hour': 'N/A',
                'peak_usage': 0.0,
                'total_predicted': 0.0
            }
        
        conn.close()
        
        return render_template('forecast.html', 
                             forecast_data=json.dumps(forecast_chart_data),
                             forecast_summary=forecast_summary)
    
    except Exception as e:
        flash(f'Error generating forecast: {str(e)}', 'error')
        return render_template('forecast.html', 
                             forecast_data=json.dumps({}),
                             forecast_summary={})

@app.route('/optimization')
@login_required
def optimization():
    """Optimization page with suggestions to reduce power usage"""
    try:
        conn = get_db_connection()
        
        # Get recent data for optimization
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 168", 
            conn, params=[session['user_id']]
        )
        
        # Convert timestamp strings back to datetime for processing
        if len(recent_data) > 0 and 'timestamp' in recent_data.columns:
            recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
        
        if len(recent_data) > 0:
            try:
                # Get optimization suggestions
                suggestions = optimization_engine.get_optimization_suggestions(recent_data)
                
                # Calculate potential savings
                current_cost = optimization_engine.calculate_cost(recent_data)
                optimized_cost = optimization_engine.calculate_optimized_cost(recent_data)
                potential_savings = current_cost - optimized_cost
                
                # Get usage patterns
                usage_patterns = optimization_engine.analyze_usage_patterns(recent_data)
            except Exception as opt_error:
                print(f"Optimization error: {str(opt_error)}")
                # Fallback to empty results
                suggestions = []
                current_cost = 0
                optimized_cost = 0
                potential_savings = 0
                usage_patterns = {}
        else:
            # New user or no data - start with zeros
            suggestions = []
            current_cost = 0
            optimized_cost = 0
            potential_savings = 0
            usage_patterns = {}
        
        conn.close()
        
        # Sanitize for JSON
        safe_suggestions = sanitize_for_json(suggestions)
        safe_usage_patterns = sanitize_for_json(usage_patterns)
        # Build optimization_data for the JS chart, always provide all keys
        optimization_data = {
            'peak_hours': safe_usage_patterns.get('peak_hours', ['6AM', '9AM', '12PM', '3PM', '6PM', '9PM']),
            'current_usage': safe_usage_patterns.get('current_usage', [2.1, 3.2, 2.8, 3.5, 4.2, 3.8]),
            'optimized_usage': safe_usage_patterns.get('optimized_usage', [1.8, 2.9, 2.5, 3.1, 3.7, 3.3]),
        }
        return render_template('optimization.html',
                             suggestions=safe_suggestions,
                             current_cost=current_cost,
                             optimized_cost=optimized_cost,
                             potential_savings=potential_savings,
                             usage_patterns=safe_usage_patterns,
                             optimization_data=optimization_data)
    
    except Exception as e:
        flash(f'Error generating optimization suggestions: {str(e)}', 'error')
        return render_template('optimization.html',
                             suggestions=[],
                             current_cost=0,
                             optimized_cost=0,
                             potential_savings=0,
                             usage_patterns={})

@app.route('/upload_data', methods=['POST'])
@login_required
def upload_data():
    """Handle data upload from CSV files"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'})
        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Ensure upload directory exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            # Debug: Print file information
            print(f"Upload: Filename: {filename}")
            print(f"Upload: Filepath: {filepath}")
            print(f"Upload: Upload folder: {app.config['UPLOAD_FOLDER']}")
            print(f"Upload: File exists before save: {os.path.exists(filepath)}")
            print(f"Upload: User ID: {session.get('user_id')}")
            
            file.save(filepath)
            
            # Debug: Check if file was saved
            print(f"Upload: File exists after save: {os.path.exists(filepath)}")
            print(f"Upload: File size: {os.path.getsize(filepath) if os.path.exists(filepath) else 'File not found'}")
            
            # Check file permissions
            try:
                import stat
                file_stat = os.stat(filepath)
                print(f"Upload: File permissions: {oct(file_stat.st_mode)}")
                print(f"Upload: File is readable: {os.access(filepath, os.R_OK)}")
                print(f"Upload: File is writable: {os.access(filepath, os.W_OK)}")
            except Exception as perm_error:
                print(f"Upload: Permission check error: {str(perm_error)}")
            
            # Process the uploaded data with encoding detection
            file_extension = filename.rsplit('.', 1)[1].lower()
            
            if file_extension in ['xlsx', 'xls']:
                # Handle Excel files
                print(f"Upload: Attempting to read Excel file: {filepath}")
                print(f"Upload: File exists before reading: {os.path.exists(filepath)}")
                try:
                    df = pd.read_excel(filepath)
                    print(f"Upload: Successfully read Excel file, shape: {df.shape}")
                except ImportError as e:
                    print(f"Upload: ImportError: {str(e)}")
                    if 'openpyxl' in str(e):
                        return jsonify({'success': False, 'message': 'Excel file support requires openpyxl. Please run: pip install openpyxl'})
                    elif 'xlrd' in str(e):
                        return jsonify({'success': False, 'message': 'Excel file support requires xlrd. Please run: pip install xlrd'})
                    else:
                        return jsonify({'success': False, 'message': f'Missing Excel dependency: {str(e)}'})
                except Exception as e:
                    print(f"Upload: Exception reading Excel file: {str(e)}")
                    return jsonify({'success': False, 'message': f'Error reading Excel file: {str(e)}'})
            else:
                # Handle CSV files with encoding detection
                print(f"Upload: Attempting to read CSV file: {filepath}")
                print(f"Upload: File exists before reading: {os.path.exists(filepath)}")
                try:
                    detected_encoding = detect_file_encoding(filepath)
                    print(f"Upload: Detected encoding: {detected_encoding}")
                    df = pd.read_csv(filepath, encoding=detected_encoding)
                    print(f"Upload: Successfully read CSV file, shape: {df.shape}")
                except Exception as e:
                    print(f"Upload: Exception reading CSV file: {str(e)}")
                    return jsonify({'success': False, 'message': f'Error reading CSV file: {str(e)}'})
            
            # Validate and process the data
            print(f"Upload: DataFrame columns: {df.columns.tolist()}")
            print(f"Upload: DataFrame shape: {df.shape}")
            print(f"Upload: DataFrame head:\n{df.head()}")
            try:
                processed_data = data_processor.process_uploaded_data(df)
                print(f"Upload: Successfully processed data, shape: {processed_data.shape}")
            except Exception as e:
                print(f"Upload: Exception processing data: {str(e)}")
                return jsonify({'success': False, 'message': f'Error processing data: {str(e)}'})
            
            # Save to database
            try:
                print(f"Upload: Starting database save...")
                conn = get_db_connection()
                
                # Select only the columns that exist in the database schema
                db_columns = ['timestamp', 'usage', 'temperature', 'humidity', 'appliance_usage', 'data_source']
                available_columns = [col for col in db_columns if col in processed_data.columns]
                print(f"Upload: Available columns for database: {available_columns}")
                
                # Prepare data for database insertion
                db_data = processed_data[available_columns].copy()
                
                # Add user_id to the data
                db_data['user_id'] = session.get('user_id')
                
                # Convert timestamps to strings before saving to SQLite
                if 'timestamp' in db_data.columns:
                    db_data['timestamp'] = db_data['timestamp'].astype(str)
                
                print(f"Upload: Saving {len(db_data)} records to database")
                print(f"Upload: First timestamp: {db_data['timestamp'].iloc[0]}")
                print(f"Upload: Last timestamp: {db_data['timestamp'].iloc[-1]}")
                print(f"Upload: Sample usage values: {db_data['usage'].head().tolist()}")
                print(f"Upload: User ID: {session.get('user_id')}")
                print(f"Upload: Database columns: {db_data.columns.tolist()}")
                
                print(f"Upload: About to save to database, connection state: {conn}")
                
                # Try using raw SQL instead of to_sql to avoid potential pandas/SQLite issues
                try:
                    cursor = conn.cursor()
                    
                    # Prepare INSERT statement
                    columns = db_data.columns.tolist()
                    placeholders = ', '.join(['?' for _ in columns])
                    sql = f"INSERT INTO power_usage ({', '.join(columns)}) VALUES ({placeholders})"
                    print(f"Upload: SQL statement: {sql}")
                    
                    # Insert data row by row
                    for index, row in db_data.iterrows():
                        values = [row[col] for col in columns]
                        cursor.execute(sql, values)
                    
                    print(f"Upload: Successfully inserted {len(db_data)} records using raw SQL")
                    
                except Exception as sql_error:
                    print(f"Upload: Raw SQL insert failed: {str(sql_error)}")
                    print(f"Upload: Falling back to pandas to_sql")
                    
                    # Fallback to pandas to_sql
                    db_data.to_sql('power_usage', conn, if_exists='append', index=False)
                    print(f"Upload: Successfully saved to database using pandas to_sql")
                
                conn.commit()
                print(f"Upload: Database commit successful")
                
                # Verify data was saved correctly
                try:
                    # Check if data was actually saved
                    verification_query = "SELECT COUNT(*) as count FROM power_usage WHERE user_id = ? AND data_source = 'uploaded'"
                    verification_result = conn.execute(verification_query, (session.get('user_id'),)).fetchone()
                    print(f"Upload: Verification - Found {verification_result['count']} uploaded records for user {session.get('user_id')}")
                    
                    # Check recent data
                    recent_query = "SELECT * FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5"
                    recent_data = pd.read_sql_query(recent_query, conn, params=[session.get('user_id')])
                    print(f"Upload: Recent data for user: {len(recent_data)} records")
                    if len(recent_data) > 0:
                        print(f"Upload: Most recent timestamp: {recent_data.iloc[0]['timestamp']}")
                        print(f"Upload: Most recent usage: {recent_data.iloc[0]['usage']}")
                        print(f"Upload: Most recent data source: {recent_data.iloc[0]['data_source']}")
                    
                except Exception as verify_error:
                    print(f"Upload: Verification error: {str(verify_error)}")
                
                conn.close()
                print(f"Upload: Database connection closed")
                
                # Clean up uploaded file
                print(f"Upload: Cleaning up file: {filepath}")
                print(f"Upload: File exists before cleanup: {os.path.exists(filepath)}")
                try:
                    os.remove(filepath)
                    print(f"Upload: File exists after cleanup: {os.path.exists(filepath)}")
                except Exception as cleanup_error:
                    print(f"Upload: File cleanup error: {str(cleanup_error)}")
                
                return jsonify({'success': True, 'message': f'Data uploaded successfully! {len(processed_data)} records added.'})
            except Exception as e:
                print(f"Upload: Database save exception: {str(e)}")
                print(f"Upload: Exception type: {type(e)}")
                import traceback
                print(f"Upload: Traceback: {traceback.format_exc()}")
                return jsonify({'success': False, 'message': f'Error saving to database: {str(e)}'})
        else:
            return jsonify({'success': False, 'message': 'Invalid file format. Please upload a CSV, XLSX, or XLS file.'})
    
    except Exception as e:
        print(f"Upload: Main exception: {str(e)}")
        print(f"Upload: Exception type: {type(e)}")
        import traceback
        print(f"Upload: Main traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Error uploading data: {str(e)}'})

@app.route('/api/current_usage')
def api_current_usage():
    """API endpoint to get current power usage"""
    try:
        conn = get_db_connection()
        result = conn.execute(
            "SELECT usage FROM power_usage ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        
        if result:
            return jsonify({'usage': result[0], 'timestamp': datetime.now().isoformat()})
        else:
            return jsonify({'usage': 0, 'timestamp': datetime.now().isoformat()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast_data')
def api_forecast_data():
    """API endpoint to get forecast data"""
    try:
        conn = get_db_connection()
        historical_data = pd.read_sql_query(
            "SELECT * FROM power_usage ORDER BY timestamp DESC LIMIT 168", 
            conn
        )
        
        # Convert timestamp strings back to datetime for processing
        if len(historical_data) > 0 and 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        conn.close()
        
        if len(historical_data) > 0:
            processed_data = data_processor.prepare_for_forecasting(historical_data)
            forecast_data = forecasting_model.predict(processed_data)
            return jsonify(forecast_data)
        else:
            return jsonify({'error': 'No data available for forecasting'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast')
def api_forecast():
    """API endpoint to get forecast data for a given period"""
    try:
        hours = int(request.args.get('hours', 24))
        conn = get_db_connection()
        historical_data = pd.read_sql_query(
            "SELECT * FROM power_usage ORDER BY timestamp DESC LIMIT 168", 
            conn
        )
        if len(historical_data) > 0 and 'timestamp' in historical_data.columns:
            historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        conn.close()
        if len(historical_data) > 0:
            processed_data = data_processor.prepare_for_forecasting(historical_data)
            forecast_data = forecasting_model.predict(processed_data)
            # Only return the requested number of hours
            result = {
                "timestamps": forecast_data.get("timestamps", [])[:hours],
                "forecast": forecast_data.get("predictions", [])[:hours],
                "confidence_upper": forecast_data.get("confidence_upper", [])[:hours],
                "confidence_lower": forecast_data.get("confidence_lower", [])[:hours],
            }
            return jsonify({"success": True, "data": result})
        else:
            return jsonify({"success": False, "message": "No data available for forecasting"}), 400
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/optimization_suggestions')
def api_optimization_suggestions():
    """API endpoint to get optimization suggestions"""
    try:
        conn = get_db_connection()
        # Since this is an API endpoint, we need to check if user is authenticated
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 168", 
            conn, params=[session['user_id']]
        )
        
        # Convert timestamp strings back to datetime for processing
        if len(recent_data) > 0 and 'timestamp' in recent_data.columns:
            recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
        
        conn.close()
        
        if len(recent_data) > 0:
            suggestions = optimization_engine.get_optimization_suggestions(recent_data)
            return jsonify(suggestions)
        else:
            return jsonify({'error': 'No data available for optimization'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_sample_data', methods=['POST'])
@login_required
def api_generate_sample_data():
    """API endpoint to generate sample data"""
    try:
        generate_sample_data(user_id=session['user_id'])
        return jsonify({'success': True, 'message': 'Sample data generated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard_data')
@login_required
def api_dashboard_data():
    """API endpoint to get dashboard data for dynamic updates"""
    try:
        conn = get_db_connection()
        
        # Get recent power usage data from the last 7 days
        from datetime import datetime, timedelta
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        # Since @login_required ensures user_id is in session, use it directly
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE timestamp >= ? AND user_id = ? ORDER BY timestamp DESC", 
            conn, params=[seven_days_ago, session['user_id']]
        )
        
        # Convert timestamp strings back to datetime for processing
        if len(recent_data) > 0 and 'timestamp' in recent_data.columns:
            recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
        
        # Get current statistics
        if len(recent_data) > 0:
            current_usage = float(recent_data.iloc[0]['usage'])
            avg_usage = float(recent_data['usage'].mean())
            total_usage = float(recent_data['usage'].sum())
            
            # Prepare data for charts (show more data to include uploaded data)
            # Since data is ordered DESC, we want the first 100 records (most recent)
            chart_data = {
                'timestamps': recent_data['timestamp'].head(100).dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                'usage': [float(x) for x in recent_data['usage'].head(100).tolist()],
                'temperature': [float(x) for x in recent_data['temperature'].head(100).tolist()]
            }
        else:
            # New user or no data - start with zeros
            current_usage = 0.0
            avg_usage = 0.0
            total_usage = 0.0
            chart_data = {
                'timestamps': [],
                'usage': [],
                'temperature': []
            }
        
        conn.close()
        
        return jsonify({
            'success': True,
            'current_usage': current_usage,
            'avg_usage': avg_usage,
            'total_usage': total_usage,
            'chart_data': chart_data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/debug_data')
def api_debug_data():
    """API endpoint to debug database contents"""
    try:
        conn = get_db_connection()
        
        # Get total count
        total_count = conn.execute("SELECT COUNT(*) FROM power_usage").fetchone()[0]
        
        # Get recent data
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage ORDER BY timestamp DESC LIMIT 10", 
            conn
        )
        
        # Get data by source
        source_counts = pd.read_sql_query(
            "SELECT data_source, COUNT(*) as count FROM power_usage GROUP BY data_source", 
            conn
        )
        
        # Get data from last 7 days
        from datetime import datetime, timedelta
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_7_days = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE timestamp >= ? ORDER BY timestamp DESC", 
            conn, params=[seven_days_ago]
        )
        
        conn.close()
        
        return jsonify({
            'total_records': total_count,
            'recent_data': recent_data.to_dict('records') if len(recent_data) > 0 else [],
            'source_counts': source_counts.to_dict('records') if len(source_counts) > 0 else [],
            'recent_7_days_count': len(recent_7_days),
            'recent_7_days_data': recent_7_days.to_dict('records') if len(recent_7_days) > 0 else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug_user_data')
@login_required
def api_debug_user_data():
    """API endpoint to debug user-specific database contents"""
    try:
        conn = get_db_connection()
        
        # Get total count for this user
        total_count = conn.execute("SELECT COUNT(*) FROM power_usage WHERE user_id = ?", (session['user_id'],)).fetchone()[0]
        
        # Get recent data for this user
        recent_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10", 
            conn, params=[session['user_id']]
        )
        
        # Get data by source for this user
        source_counts = pd.read_sql_query(
            "SELECT data_source, COUNT(*) as count FROM power_usage WHERE user_id = ? GROUP BY data_source", 
            conn, params=[session['user_id']]
        )
        
        # Get data from last 7 days for this user
        from datetime import datetime, timedelta
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_7_days = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE timestamp >= ? AND user_id = ? ORDER BY timestamp DESC", 
            conn, params=[seven_days_ago, session['user_id']]
        )
        
        # Get dashboard query data (same as dashboard route)
        dashboard_data = pd.read_sql_query(
            "SELECT * FROM power_usage WHERE timestamp >= ? AND user_id = ? ORDER BY timestamp DESC", 
            conn, params=[seven_days_ago, session['user_id']]
        )
        
        conn.close()
        
        return jsonify({
            'user_id': session['user_id'],
            'total_records': total_count,
            'recent_data': recent_data.to_dict('records') if len(recent_data) > 0 else [],
            'source_counts': source_counts.to_dict('records') if len(source_counts) > 0 else [],
            'recent_7_days_count': len(recent_7_days),
            'recent_7_days_data': recent_7_days.to_dict('records') if len(recent_7_days) > 0 else [],
            'dashboard_data_count': len(dashboard_data),
            'dashboard_data': dashboard_data.to_dict('records') if len(dashboard_data) > 0 else [],
            'seven_days_ago': seven_days_ago.isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear_all_data', methods=['POST'])
def clear_all_data():
    """Delete all records from the power_usage table"""
    try:
        conn = get_db_connection()
        conn.execute('DELETE FROM power_usage')
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'All data cleared.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/settings')
@login_required
def settings():
    """Settings page for user preferences"""
    try:
        conn = get_db_connection()
        
        # Get user settings
        settings_data = conn.execute(
            'SELECT key, value FROM settings WHERE user_id = ?',
            (session['user_id'],)
        ).fetchall()
        
        # Convert to dictionary
        settings = {row['key']: row['value'] for row in settings_data}
        
        conn.close()
        
        # If user has no settings, create default ones
        if not settings:
            from utils.database import create_default_settings
            create_default_settings(session['user_id'])
            
            # Get the default settings
            conn = get_db_connection()
            settings_data = conn.execute(
                'SELECT key, value FROM settings WHERE user_id = ?',
                (session['user_id'],)
            ).fetchall()
            settings = {row['key']: row['value'] for row in settings_data}
            conn.close()
        
        return render_template('settings.html', settings=settings)
    except Exception as e:
        flash(f'Error loading settings: {str(e)}', 'error')
        return render_template('settings.html', settings={})

@app.route('/update_settings', methods=['POST'])
@login_required
def update_settings():
    """Handle settings updates"""
    try:
        # Get form data
        electricity_rate = float(request.form.get('electricity_rate', 0.12))
        green_energy = request.form.get('green_energy') == 'on'
        notifications = request.form.get('notifications') == 'on'
        peak_hours = request.form.get('peak_hours', '18:00-22:00')
        high_usage_threshold = float(request.form.get('high_usage_threshold', 4.0))
        alert_frequency = request.form.get('alert_frequency', 'weekly')
        
        # Save settings to database
        conn = get_db_connection()
        
        # Delete existing settings for this user
        conn.execute('DELETE FROM settings WHERE user_id = ?', (session['user_id'],))
        
        # Insert new settings
        settings_data = [
            ('electricity_rate', str(electricity_rate), session['user_id']),
            ('green_energy', str(green_energy), session['user_id']),
            ('notifications', str(notifications), session['user_id']),
            ('peak_hours', peak_hours, session['user_id']),
            ('high_usage_threshold', str(high_usage_threshold), session['user_id']),
            ('alert_frequency', alert_frequency, session['user_id'])
        ]
        
        conn.executemany(
            'INSERT INTO settings (key, value, user_id) VALUES (?, ?, ?)',
            settings_data
        )
        conn.commit()
        conn.close()
        
        flash('Settings updated successfully!', 'success')
    
    except Exception as e:
        flash(f'Error updating settings: {str(e)}', 'error')
    
    return redirect(url_for('settings'))

@app.route('/api/clear_user_data', methods=['POST'])
@login_required
def clear_user_data():
    """Clear all data for the current user"""
    try:
        conn = get_db_connection()
        
        # Clear power usage data
        conn.execute('DELETE FROM power_usage WHERE user_id = ?', (session['user_id'],))
        
        # Clear forecasts
        conn.execute('DELETE FROM forecasts WHERE user_id = ?', (session['user_id'],))
        
        # Clear optimization suggestions
        conn.execute('DELETE FROM optimization_suggestions WHERE user_id = ?', (session['user_id'],))
        
        # Clear settings
        conn.execute('DELETE FROM settings WHERE user_id = ?', (session['user_id'],))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'All user data cleared successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/delete_account', methods=['POST'])
@login_required
def delete_account():
    """Delete the current user's account and all associated data"""
    try:
        conn = get_db_connection()
        
        # Clear all user data first
        conn.execute('DELETE FROM power_usage WHERE user_id = ?', (session['user_id'],))
        conn.execute('DELETE FROM forecasts WHERE user_id = ?', (session['user_id'],))
        conn.execute('DELETE FROM optimization_suggestions WHERE user_id = ?', (session['user_id'],))
        conn.execute('DELETE FROM settings WHERE user_id = ?', (session['user_id'],))
        
        # Delete the user account
        conn.execute('DELETE FROM users WHERE id = ?', (session['user_id'],))
        
        conn.commit()
        conn.close()
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Account deleted successfully'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_data_retention', methods=['POST'])
@login_required
def update_data_retention():
    """Update data retention settings"""
    try:
        data = request.get_json()
        retention_days = int(data.get('retention_days', 0))
        
        if retention_days > 0:
            # Calculate cutoff date
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = get_db_connection()
            
            # Delete old data
            conn.execute(
                'DELETE FROM power_usage WHERE user_id = ? AND timestamp < ?',
                (session['user_id'], cutoff_date)
            )
            
            conn.execute(
                'DELETE FROM forecasts WHERE user_id = ? AND timestamp < ?',
                (session['user_id'], cutoff_date)
            )
            
            conn.commit()
            conn.close()
            
            return jsonify({'success': True, 'message': f'Data older than {retention_days} days has been removed'})
        else:
            return jsonify({'success': True, 'message': 'All data will be retained'})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information"""
    try:
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password:
            # Verify current password
            conn = get_db_connection()
            user = conn.execute(
                'SELECT password_hash FROM users WHERE id = ?',
                (session['user_id'],)
            ).fetchone()
            
            if not user or not verify_password(current_password, user['password_hash']):
                flash('Current password is incorrect.', 'error')
                return redirect(url_for('profile'))
            
            if new_password != confirm_password:
                flash('New passwords do not match.', 'error')
                return redirect(url_for('profile'))
            
            if len(new_password) < 6:
                flash('New password must be at least 6 characters long.', 'error')
                return redirect(url_for('profile'))
            
            # Update password
            hashed_password = hash_password(new_password)
            conn.execute(
                'UPDATE users SET password_hash = ? WHERE id = ?',
                (hashed_password, session['user_id'])
            )
            conn.commit()
            conn.close()
            
            flash('Password updated successfully!', 'success')
        else:
            flash('No changes made.', 'info')
    
    except Exception as e:
        flash(f'Error updating profile: {str(e)}', 'error')
    
    return redirect(url_for('profile'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)