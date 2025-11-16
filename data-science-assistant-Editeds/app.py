import streamlit as st
from tools.dataset_tools import DatasetManager
from agents.planner import Planner
from agents.worker import Worker
from core.entropy_logger import EntropyLogger  # Added
import os
import pandas as pd
import sqlite3
import time  # Added for execution time tracking

st.set_page_config(page_title="Data Science Assistant", layout="wide")
st.title("Data Science Assistant")

# Initialize logger
logger = EntropyLogger()

# Initialize agents
planner = Planner()
worker = Worker()

# Dataset management section
st.sidebar.header("📁 Dataset Management")
dm = DatasetManager()

# Option 1: Load from predefined datasets
st.sidebar.subheader("Load Example Dataset")
if st.sidebar.button("Load Titanic Dataset"):
    try:
        if os.path.exists("datasets/titanic.csv"):
            success = dm.load_from_path("titanic", "datasets/titanic.csv")
            if success:
                st.sidebar.success("✓ Titanic dataset loaded!")
            else:
                st.sidebar.error("Failed to load Titanic dataset")
        else:
            st.sidebar.warning("Titanic dataset file not found")
    except Exception as e:
        st.sidebar.error(f"Error loading Titanic dataset: {e}")

# Option 2: Upload custom dataset
st.sidebar.subheader("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file",
    type="csv",
    help="Upload a CSV file for analysis"
)

if uploaded_file is not None:
    success = dm.load_from_upload(uploaded_file)
    if success:
        st.sidebar.success(f"✓ {uploaded_file.name} loaded successfully!")

# Display dataset info
st.sidebar.subheader("Dataset Info")
if dm.get_dataset() is not None:
    dataset_info = dm.get_dataset_info()
    st.sidebar.write(f"**Name:** {dataset_info['name']}")
    st.sidebar.write(f"**Shape:** {dataset_info['shape']}")
    st.sidebar.write(f"**Columns:** {', '.join(dataset_info['columns'])}")
else:
    st.sidebar.info("No dataset loaded. Please load or upload a dataset.")

# Display current mode
st.sidebar.subheader("Current Mode")
planning_mode = planner.get_planning_mode() if hasattr(planner, 'get_planning_mode') else "Unknown"
st.sidebar.info(f"📋 Planning: {planning_mode}")

if hasattr(worker, 'use_openai'):
    code_gen_mode = "OpenAI" if worker.use_openai else "Qwen"
    st.sidebar.info(f"💻 Code Generation: {code_gen_mode}")

# Entropy Metrics in Sidebar
st.sidebar.header("📈 Entropy Metrics")
if st.sidebar.button("Show Entropy Report"):
    try:
        metrics = logger.get_entropy_metrics()
        
        if "error" in metrics:
            st.sidebar.error(metrics["error"])
        else:
            st.sidebar.success("✅ Entropy Tracking Active")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Total Queries", metrics['total_interactions'])
                st.metric("Success Rate", f"{metrics['success_rate']*100:.1f}%")
            with col2:
                st.metric("Avg Time", f"{metrics['average_fix_time']:.2f}s")
                st.metric("Total Errors", metrics['total_errors'])
    except Exception as e:
        st.sidebar.error(f"Could not load entropy metrics: {e}")

# Instructions
st.sidebar.header("ℹ️ Instructions")
st.sidebar.info("""
1. Load a dataset using the sidebar options
2. Ask a data science question in natural language
3. The assistant will create and execute a plan
4. Results will include plots, tables, and code
""")

st.sidebar.header("💡 Example Queries")
st.sidebar.text("""
- Plot survival rate by Sex
- Show average age by class
- Analyze correlation between features
- Create histogram of ages
- Show missing values
- Group by embarked and show stats
""")

# --- Tabs for navigation ---
tab1, tab2 = st.tabs(["🔍 Data Analysis", "📊 Entropy Metrics"])

# ---------------------------
# TAB 1: Data Analysis
# ---------------------------
with tab1:

    if dm.get_dataset() is None:
        st.warning("Please load a dataset first using the sidebar options.")
    else:
        st.subheader("📊 Dataset Preview")
        st.dataframe(dm.get_dataset().head())

        user_query = st.text_input(
            "Ask a data science question:",
            placeholder="e.g., 'Plot survival rate by Sex', 'Show average age by class'",
            key="query_input"
        )

        if st.button("Run Analysis", type="primary") and user_query.strip():
            start_time = time.time()  # Start timing
            
            with st.spinner("Planning and executing..."):
                try:
                    # Create plan
                    plan = planner.plan(user_query)

                    # Fallback for missing 'steps' with Worker-compatible dicts
                    if 'steps' not in plan or not plan['steps']:
                        st.warning(
                            "Planner did not generate explicit steps. Using analysis and visualization sections as fallback."
                        )
                        pseudo_steps = []
                        for a in plan.get('analysis', []):
                            pseudo_steps.append({"action": a, "params": {}})
                        for v in plan.get('visualization', []):
                            pseudo_steps.append({"action": v, "params": {}})
                        plan['steps'] = pseudo_steps

                    # Critique plan (optional)
                    if hasattr(planner, 'critique'):
                        critique = planner.critique(plan)
                        plan["self_check"] = critique

                    st.subheader("📋 Analysis Plan (with Self-Critique):")
                    st.json(plan)

                    # Execute plan safely
                    result = worker.handle(plan, dm) if hasattr(worker, 'handle') else {}

                    execution_time = time.time() - start_time  # Calculate execution time
                    
                    # Log successful execution
                    logger.log_interaction(
                        prompt_text=user_query, 
                        response_text="Analysis completed successfully", 
                        error_count=0, 
                        fix_time=execution_time
                    )

                    st.subheader("📈 Results:")

                    # --- Display images safely ---
                    for img_path in result.get("images", []):
                        if os.path.exists(img_path):
                            st.image(img_path)
                        else:
                            st.warning(f"Image file not found: {img_path}")

                    # --- Display tables safely ---
                    for i, table in enumerate(result.get("tables", [])):
                        if isinstance(table, (pd.DataFrame, pd.Series)):
                            st.dataframe(table)
                        else:
                            st.write(table)

                    # --- Display execution details safely ---
                    for action, action_result in result.get("results", {}).items():
                        with st.expander(f"Action: {action}"):
                            if action_result.get("success"):
                                st.success("✅ Execution successful")

                                if action_result.get("warnings"):
                                    st.warning("⚠️ Warnings detected:")
                                    for warning in action_result["warnings"]:
                                        st.text(warning)

                                if action_result.get("print_output"):
                                    st.subheader("Console Output:")
                                    for output in action_result["print_output"]:
                                        st.text(output)

                                if "output" in action_result:
                                    st.write("Output:", action_result["output"])

                                st.code(action_result.get("code", ""), language="python")
                            else:
                                st.error(f"❌ Error: {action_result.get('error', 'Unknown error')}")
                                st.code(action_result.get("code", ""), language="python")

                    # Refresh sidebar info
                    planning_mode = planner.get_planning_mode() if hasattr(planner, 'get_planning_mode') else "Unknown"
                    st.sidebar.info(f"📋 Planning: {planning_mode}")

                    if hasattr(worker, 'use_openai'):
                        code_gen_mode = "OpenAI" if worker.use_openai else "Qwen"
                        st.sidebar.info(f"💻 Code Generation: {code_gen_mode}")

                except Exception as e:
                    execution_time = time.time() - start_time  # Calculate execution time even for errors
                    
                    # Log error with execution time
                    logger.log_interaction(
                        prompt_text=user_query, 
                        response_text=f"Error: {str(e)}", 
                        error_count=1, 
                        fix_time=execution_time
                    )
                    
                    st.error(f"Error during analysis: {str(e)}")

# ---------------------------
# TAB 2: Entropy Metrics
# ---------------------------
with tab2:
    st.header("📊 Entropy Metrics Dashboard")

    if os.path.exists("entropy_log.db"):
        try:
            # Use the logger's get_logs method which includes compatibility columns
            df = logger.get_logs(limit=1000)  # Get all logs
            
            if not df.empty:
                st.subheader("📈 System Performance Overview")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_interactions = len(df)
                    st.metric("Total Interactions", total_interactions)
                
                with col2:
                    success_rate = (df['error_count'] == 0).mean() * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col3:
                    avg_time = df['fix_time'].mean()
                    st.metric("Avg Execution Time", f"{avg_time:.2f}s")
                
                with col4:
                    total_errors = df['error_count'].sum()
                    st.metric("Total Errors", total_errors)

                st.subheader("📋 Recent Logs")
                # Show most recent logs with important columns
                display_columns = ['timestamp', 'prompt', 'error_count', 'fix_time', 'success']
                available_columns = [col for col in display_columns if col in df.columns]
                
                if available_columns:
                    st.dataframe(df[available_columns].head(10))
                else:
                    st.dataframe(df.head(10))

                # Time series analysis
                st.subheader("📊 Activity Over Time")
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    daily_activity = df.groupby(df['timestamp'].dt.date).size()
                    st.line_chart(daily_activity)

                # Error analysis
                st.subheader("🔍 Error Analysis")
                error_df = df[df['error_count'] > 0]
                if not error_df.empty:
                    st.write(f"**Error Patterns:** {len(error_df)} queries had errors")
                    
                    # Show error-prone queries
                    if 'prompt' in error_df.columns:
                        st.write("**Recent queries with errors:**")
                        for _, row in error_df.head(5).iterrows():
                            st.write(f"- {row['prompt'][:80]}... (Errors: {row['error_count']})")
                else:
                    st.success("🎉 No errors in the recorded logs!")
                        
            else:
                st.info("No entropy data recorded yet. Start interacting with the assistant to create logs.")

        except Exception as e:
            st.error(f"Could not read entropy logs: {e}")
            st.info("Trying alternative method...")
            
            # Fallback to direct SQLite access
            try:
                conn = sqlite3.connect("entropy_log.db")
                df_raw = pd.read_sql_query("SELECT * FROM logs", conn)
                conn.close()
                
                if not df_raw.empty:
                    st.subheader("Raw Database Logs")
                    st.dataframe(df_raw)
                else:
                    st.info("No data in entropy database.")
                    
            except Exception as sql_error:
                st.error(f"Could not access database directly: {sql_error}")
                
    else:
        st.info("Entropy log database not found. Start interacting with the assistant to create logs.")
        st.info("The database will be created automatically when you run your first analysis.")