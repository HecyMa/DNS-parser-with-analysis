import threading
from api import api_run
from parser import run


if __name__ == "__main__":
    api_thread = threading.Thread(target=api_run)
    api_thread.start()  # http://localhost:8000
    run()
