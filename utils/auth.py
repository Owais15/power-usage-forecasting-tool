import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, flash, request
from utils.database import get_db_connection

def hash_password(password):
    """Hash a password using SHA-256 with salt"""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((password + salt).encode())
    return salt + hash_obj.hexdigest()

def verify_password(password, hashed_password):
    """Verify a password against its hash"""
    if len(hashed_password) < 32:  # Salt is 32 chars (16 bytes as hex)
        return False
    salt = hashed_password[:32]
    hash_obj = hashlib.sha256((password + salt).encode())
    return hashed_password[32:] == hash_obj.hexdigest()

def create_user(username, email, password):
    """Create a new user account"""
    conn = get_db_connection()
    try:
        # Check if username or email already exists
        existing_user = conn.execute(
            'SELECT id FROM users WHERE username = ? OR email = ?',
            (username, email)
        ).fetchone()
        
        if existing_user:
            return False, "Username or email already exists"
        
        # Create new user
        hashed_password = hash_password(password)
        conn.execute(
            'INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)',
            (username, email, hashed_password, datetime.now())
        )
        conn.commit()
        return True, "User created successfully"
    except Exception as e:
        return False, f"Error creating user: {str(e)}"
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate a user and return user data if successful"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT id, username, email, password_hash FROM users WHERE username = ? OR email = ?',
            (username, username)
        ).fetchone()
        
        if user and verify_password(password, user['password_hash']):
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        return None
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None
    finally:
        conn.close()

def get_user_by_id(user_id):
    """Get user information by ID"""
    conn = get_db_connection()
    try:
        user = conn.execute(
            'SELECT id, username, email, created_at FROM users WHERE id = ?',
            (user_id,)
        ).fetchone()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error getting user: {str(e)}")
        return None
    finally:
        conn.close()

def login_required(f):
    """Decorator to require login for protected routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def logout_user():
    """Log out the current user"""
    session.clear()

def is_authenticated():
    """Check if user is currently authenticated"""
    return 'user_id' in session

def get_current_user():
    """Get current authenticated user data"""
    if 'user_id' in session:
        return get_user_by_id(session['user_id'])
    return None
