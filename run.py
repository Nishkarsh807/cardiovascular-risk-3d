"""
CardioVision 3D - Application Launch Entrypoint.
Starts the FastAPI server and optionally opens the browser.
"""

import sys
import os
import webbrowser
import uvicorn

def main():
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f" CardioVision 3D - Cardiovascular Risk & Stenosis System")
    print(f" Multimodal AI Hackathon 2026 (Track A)")
    print(f"=======================================================")
    print(f" Server starting at: {url}")
    print(f" Interactive 3D Web App: {url}")
    print(f" API Documentation: {url}/docs")
    print(f" Press Ctrl+C to terminate the server.\n")

    # Open browser automatically if not in headless environment
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("src.api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
