# logs/simple_logger.py
import csv, os, time

LOG_PATH = "logs/entropy_log.csv"

def log_event(event_type: str, detail: str = ""):
    os.makedirs("logs", exist_ok=True)
    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "event_type", "detail"])
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_type, detail])
