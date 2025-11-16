import sqlite3
import datetime
import pandas as pd

class EntropyLogger:
    def __init__(self, db_path="entropy_log.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            prompt_text TEXT,
            response_text TEXT,
            error_count INTEGER,
            fix_time REAL
        )""")
        self.conn.commit()

    def log_interaction(self, prompt_text, response_text=None, error_count=0, fix_time=0.0):
        """Log one complete interaction: prompt, response, errors, and debugging time."""
        ts = datetime.datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO logs (timestamp, prompt_text, response_text, error_count, fix_time) VALUES (?, ?, ?, ?, ?)",
            (ts, prompt_text, response_text, error_count, fix_time)
        )
        self.conn.commit()

    def get_logs(self, limit=50):
        """Fetch recent logs as a pandas DataFrame."""
        logs = pd.read_sql_query(
            "SELECT * FROM logs ORDER BY id DESC LIMIT ?",
            self.conn,
            params=(limit,)
        )

        # ✅ FIX: Ensure 'prompt' column exists for compatibility
        if 'prompt_text' in logs.columns and 'prompt' not in logs.columns:
            logs['prompt'] = logs['prompt_text']
            
        # ✅ Also add other expected columns
        if 'response_text' in logs.columns and 'response' not in logs.columns:
            logs['response'] = logs['response_text']
            
        if 'error_count' in logs.columns and 'errors' not in logs.columns:
            logs['errors'] = logs['error_count'].apply(lambda x: [] if x == 0 else ['Unknown error'])
            
        if 'fix_time' in logs.columns and 'execution_time' not in logs.columns:
            logs['execution_time'] = logs['fix_time']
            
        if 'success' not in logs.columns:
            logs['success'] = logs['error_count'] == 0

        return logs
    
    def get_entropy_metrics(self):
        """Get basic metrics directly from database"""
        logs_df = self.get_logs(limit=1000)
        
        if logs_df.empty:
            return {"error": "No logs available"}
        
        return {
            "total_interactions": len(logs_df),
            "success_rate": (logs_df['error_count'] == 0).mean(),
            "average_fix_time": logs_df['fix_time'].mean(),
            "total_errors": logs_df['error_count'].sum()
        }