#!/usr/bin/env python3
"""
Son of Andrew AI - Consciousness System Launcher

Simple script to start the consciousness API server.
"""

import subprocess
import sys
import os

def main():
    """Launch the consciousness API server"""
    print("🧠 Starting Son of Andrew AI Consciousness System...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("consciousness_api.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected. Consider activating it:")
        print("   source venv/bin/activate")
        print()
    
    try:
        print("🚀 Launching consciousness API on http://localhost:8000")
        print("   - Chat endpoint: POST /api/chat")
        print("   - Dream endpoint: POST /api/consciousness/dream")
        print("   - Status endpoint: GET /api/consciousness/status")
        print("   - Docs: http://localhost:8000/docs")
        print()
        print("💡 Frontend should be available at http://localhost:3000")
        print("   (Run 'cd frontend && npm run dev' in another terminal)")
        print()
        print("Press Ctrl+C to stop the server...")
        print("-" * 50)
        
        # Run uvicorn with the consciousness API
        subprocess.run([
            "uvicorn", 
            "consciousness_api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
        
    except KeyboardInterrupt:
        print("\n🔌 Consciousness system shutting down...")
    except FileNotFoundError:
        print("❌ uvicorn not found. Please install requirements:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting consciousness system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 