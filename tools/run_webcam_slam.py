from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PORT = 8765


def _port_busy(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def stop_previous_map_lab(port: int = PORT) -> None:
    """If an older Map Lab server still holds the port (e.g. started from another window), stop it.
    Only processes whose command line is this project's server are touched."""
    if not _port_busy(port) or sys.platform != "win32":
        return
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    pids = {line.split()[-1] for line in out.splitlines()
            if f"127.0.0.1:{port} " in line and "LISTENING" in line}
    for pid in pids:
        cmd = subprocess.run(["powershell", "-NoProfile", "-Command",
                              f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}').CommandLine"],
                             capture_output=True, text=True).stdout
        if "run_webcam_slam" in cmd or "app.main:app" in cmd:
            print(f"Stopping the previous Map Lab server (PID {pid}) ...")
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
        else:
            print(f"Port {port} is used by another program (PID {pid}); close it and run this again.")
            sys.exit(1)
    for _ in range(30):
        if not _port_busy(port):
            return
        time.sleep(0.2)


if __name__ == "__main__":
    from tools._deps import ensure

    ensure()
    import uvicorn

    stop_previous_map_lab()
    print(f"C270 Map Lab: http://127.0.0.1:{PORT}/slam  (Ctrl+C to stop)")
    browser = threading.Timer(3.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/slam"))
    browser.daemon = True
    browser.start()
    uvicorn.run("app.main:app", host="127.0.0.1", port=PORT)
