import pandas as pd
import streamlit as st

class DatasetManager:
    def __init__(self):
        self.current_dataset = None
        self.dataset_name = None

    def load_from_path(self, name, file_path):
        """Load dataset from file path"""
        try:
            self.current_dataset = pd.read_csv(file_path)
            self.dataset_name = name
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False

    def load_from_upload(self, uploaded_file):
        """Load dataset from uploaded file"""
        try:
            self.current_dataset = pd.read_csv(uploaded_file)
            self.dataset_name = uploaded_file.name
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False

    def get_dataset(self):
        """Get the current dataset"""
        return self.current_dataset

    def get_dataset_name(self):
        """Get the current dataset name"""
        return self.dataset_name

    def get_dataset_info(self):
        """Get information about the current dataset"""
        if self.current_dataset is None:
            return "No dataset loaded"
        
        info = {
            "name": self.dataset_name,
            "shape": self.current_dataset.shape,
            "columns": list(self.current_dataset.columns),
            "missing_values": self.current_dataset.isnull().sum().to_dict(),
            "data_types": self.current_dataset.dtypes.astype(str).to_dict()
        }
        return info

    def clear_dataset(self):
        """Clear the current dataset"""
        self.current_dataset = None
        self.dataset_name = None