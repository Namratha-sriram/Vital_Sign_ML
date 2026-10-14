"""
main.py
Entry point for PyCharm execution.
Starts the VitalSign machine learning web application.
"""

import sys
import os

if __name__ == '__main__':
    print("=========================================================")
    print("          VitalSign Machine Learning System              ")
    print("=========================================================")
    print("Launching Flask Application Server...")
    
    # Import and run app
    from app import app
    app.run(debug=True, host="127.0.0.1", port=5000)
