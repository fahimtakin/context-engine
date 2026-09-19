# run_api.py
import sys
import os
import uvicorn

# 1. Force the current file's directory straight to the top of sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

if __name__ == "__main__":
    print("--- Launching ContextEngine Containerized Production REST Gateway ---")
    
    # Pass the absolute module path layout string to let Uvicorn handle process workers cleanly
uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
