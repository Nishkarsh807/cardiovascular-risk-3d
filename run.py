"""
CardioVision 3D - Application Launch Entrypoint.
Features smart port detection, auto-healing, and browser launch.
"""

import sys
import os
import socket
import webbrowser
import requests
import uvicorn

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0

def is_cardiovision_running(port: int, host: str = "127.0.0.1") -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/api/health", timeout=1.0)
        return r.status_code == 200 and "CardioVision" in r.text
    except Exception:
        return False

def find_available_port(start_port: int = 8000, max_attempts: int = 25, host: str = "127.0.0.1") -> int:
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(p, host):
            return p
    return start_port

def main():
    host = "127.0.0.1"
    target_port = 8000

    # 1. Check if CardioVision is already running on port 8000
    if is_port_in_use(target_port, host):
        if is_cardiovision_running(target_port, host):
            url = f"http://{host}:{target_port}"
            print(f"\n=======================================================")
            print(f" CardioVision 3D is ALREADY running and active!")
            print(f" URL: {url}")
            print(f" Opening interactive 3D web application in your browser...")
            print(f"=======================================================\n")
            try:
                webbrowser.open(url)
            except Exception:
                pass
            return
        else:
            # Port 8000 is in use by another app -> find next free port
            free_port = find_available_port(target_port + 1, host=host)
            print(f"\n[Notice] Port {target_port} is busy. Automatically switching to free port: {free_port}")
            target_port = free_port

    url = f"http://{host}:{target_port}"
    print(f"\n=======================================================")
    print(f" CardioVision 3D - Cardiovascular Risk & Stenosis System")
    print(f" Multimodal AI Hackathon 2026 (Track A)")
    print(f"=======================================================")
    print(f" Server starting at: {url}")
    print(f" Interactive 3D Web App: {url}")
    print(f" API Documentation: {url}/docs")
    print(f" Press Ctrl+C to terminate the server.\n")

    # Open browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("src.api:app", host=host, port=target_port, reload=False)

if __name__ == "__main__":
    main()
