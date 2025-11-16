import streamlit as st
from agents.planner import Planner
from agents.worker import Worker
from core.entropy_logger import EntropyLogger
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

# Initialize agents and logger
planner = Planner()
worker = Worker()
logger = EntropyLogger("entropy_log.db")

st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("Data Science Assistant")

# Sidebar for customization
st.sidebar.header("Settings")
user_role = st.sidebar.selectbox("Assistant Role", ["Data Scientist", "Web-Supplement", "Advisor"])
dataset_choice = st.sidebar.selectbox("Dataset", ["Titanic-Dataset.csv"])
show_entropy = st.sidebar.checkbox("Show Entropy Metrics", True)

# User input
query = st.text_area("Ask your assistant something about the dataset:")

if st.button("Run"):
    # Log query
    logger.log_prompt(query)
    
    # Planning + execution
    plan = planner.create_plan(query)
    response = worker.execute(plan)
    
    # Log response
    logger.log_response(response)
    
    st.subheader("Assistant Response")
    st.write(response)

# Entropy Visualization
if show_entropy:
    st.subheader("Entropy Analysis")
    conn = sqlite3.connect("entropy_log.db")
    df = pd.read_sql_query("SELECT * FROM entropy", conn)
    conn.close()
    
    if not df.empty:
        st.write(df.tail(10))
        st.line_chart(df['error_count'])
    else:
        st.info("No entropy data logged yet.")