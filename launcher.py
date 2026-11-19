"""Desktop entry point. In a PyInstaller build, runs the API in-process."""
from __future__ import annotations
import os, sys, time, socket, threading, webbrowser
from pathlib import Path

def main():
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    os.chdir(root)
    port = int(os.getenv("RAG_STUDIO_PORT", "8765"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.25):
            _show_error("RAG Studio is already running", f"Port {port} is already in use. Close the other RAG Studio instance and try again.")
            return 1
    except OSError:
        pass
    try:
        import uvicorn
        import api
    except Exception as exc:
        _show_error("RAG Studio startup failed", str(exc))
        return 1
    server=uvicorn.Server(uvicorn.Config(api.app,host="127.0.0.1",port=port,log_level="warning"))
    thread=threading.Thread(target=server.run,daemon=True); thread.start()
    url=f"http://127.0.0.1:{port}"
    for _ in range(120):
        if not thread.is_alive():
            _show_error("RAG Studio startup failed", "The local API stopped during startup. Review the application logs.")
            return 1
        try:
            with socket.create_connection(("127.0.0.1",port),timeout=.25): break
        except OSError: time.sleep(.25)
    webbrowser.open(url)
    try:
        import tkinter as tk
        from tkinter import messagebox
        window=tk.Tk(); window.title("RAG Studio 1.0"); window.geometry("360x160"); window.resizable(False,False)
        tk.Label(window,text="RAG Studio is running",font=("Segoe UI",14,"bold")).pack(pady=(20,5))
        tk.Label(window,text="Your local workspace is open in your browser.").pack(pady=3)
        tk.Button(window,text="Open RAG Studio",command=lambda:webbrowser.open(url)).pack(pady=8)
        def stop():
            server.should_exit=True
            window.destroy()
        tk.Button(window,text="Stop server and exit",command=stop).pack()
        window.protocol("WM_DELETE_WINDOW",stop)
        window.mainloop()
    except Exception:
        try:
            while thread.is_alive(): time.sleep(.5)
        except KeyboardInterrupt:
            server.should_exit=True
    return 0

def _show_error(title, message):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root=tk.Tk(); root.withdraw(); messagebox.showerror(title,message); root.destroy()
    except Exception:
        print(f"{title}: {message}")

if __name__ == "__main__":
    raise SystemExit(main())
