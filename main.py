import subprocess
import sys
import time


PROCESSES = [
    ("discord", "bot.py"),
    ("telegram", "telegram_bot.py"),
]


def start_process(name: str, script: str):
    print(f"Starting {name} bot from {script}...", flush=True)
    return subprocess.Popen([sys.executable, script])


def main():
    running = {name: start_process(name, script) for name, script in PROCESSES}

    while True:
        for name, process in list(running.items()):
            code = process.poll()
            if code is not None:
                raise RuntimeError(f"{name} bot stopped with exit code {code}")
        time.sleep(5)


if __name__ == "__main__":
    main()
