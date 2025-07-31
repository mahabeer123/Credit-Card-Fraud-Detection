#!/usr/bin/env python3
"""
Credit Card Fraud Detection Demo Launcher
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Credit Card Fraud Detection Demo...")
    print("="*50)
    
    # Check if models exist
    if not os.path.exists('src/models/random_forest_model.pkl'):
        print("⚠️  Models not found! Training models first...")
        try:
            subprocess.run([sys.executable, "src/models/save_models.py"], check=True)
            print("✅ Models trained and saved!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error training models: {e}")
            return
    
    # Start Streamlit app
    print("🌐 Starting Streamlit app...")
    print("📱 Open your browser and go to: http://localhost:8501")
    print("="*50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Demo stopped. Thanks for trying it out!")
    except Exception as e:
        print(f"❌ Error starting demo: {e}")

if __name__ == "__main__":
    main() 