#!/usr/bin/env python3
"""
Database reset script to fix schema issues
"""

import os
import sqlite3
from utils.database import init_db, get_db_connection

def reset_database():
    """Reset the database and recreate with correct schema"""
    
    # Get database path
    from config import Config
    db_path = Config.DATABASE_PATH
    
    print(f"Resetting database at: {db_path}")
    
    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Removed existing database")
    
    # Create database directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Initialize new database with correct schema
    init_db()
    print("Database initialized with correct schema")
    
    # Verify the settings table structure
    conn = get_db_connection()
    cursor = conn.execute("PRAGMA table_info(settings)")
    columns = cursor.fetchall()
    
    print("\nSettings table structure:")
    for column in columns:
        print(f"  {column[1]} ({column[2]})")
    
    conn.close()
    print("\nDatabase reset completed successfully!")

if __name__ == "__main__":
    reset_database()
