#!/usr/bin/env python3
"""
Launch EarlyStrike Dashboard with Improved System Logs
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'flask': 'flask',
        'tensorflow': 'tensorflow',
        'psutil': 'psutil',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn'   # pip name -> import name
    }
    missing_packages = []
    
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies satisfied")
    return True
def main():
    print("🚀 EarlyStrike Ransomware Detection System")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return
    
    print("\n📊 Starting improved dashboard with real system logs...")
    print("🔧 Features:")
    print("   - Real system event logging")
    print("   - Actual system metrics monitoring")
    print("   - Real-time threat detection simulation")
    print("   - Improved log filtering and display")
    
    # Start the dashboard backend
    try:
        print("\n🌐 Starting web server...")
        print("   Dashboard will be available at: http://localhost:5000")
        print("   Press Ctrl+C to stop the server")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌍 Dashboard opened in browser")
            except:
                print("⚠️  Could not open browser automatically")
                print("   Please navigate to: http://localhost:5000")
        
        # Start browser thread
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Import and run the backend
        from simple_dashboard_backend import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
