# -------------------------------------------------------
# components/file_utils.py - Utility Functions for File Management
# -------------------------------------------------------

import os
import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd

# -------------------------------------------------------
# Load log or metrics data from CSV file
# -------------------------------------------------------
@st.cache_data
def load_log_data(filepath):
    """Load the log data (e.g., metrics.csv) from the specified filepath."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

# -------------------------------------------------------
# Find a .npy file inside patient-specific subfolder
# -------------------------------------------------------
def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for a .npy file inside the patient subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    return file_path if os.path.exists(file_path) else None

# -------------------------------------------------------
# Find a .npy file directly inside a directory
# -------------------------------------------------------
def find_file_in_dir(base_dir, file_name):
    """Search for a .npy file directly in the base directory."""
    file_path = os.path.join(base_dir, file_name)
    return file_path if os.path.exists(file_path) else None

# -------------------------------------------------------
# Parse filenames to group them by patient ID, region ID, and slices
# -------------------------------------------------------
def parse_filenames(files, prefix):
    """
    Parse a list of filenames and organize them into a dictionary structure:
    {patient_id: {region_id: [(filename, slice_info), ...]}}
    """
    patients = {}
    for file in files:
        if prefix in file:
            parts = file.split("_")
            patient_id = parts[0]
            region_id = parts[1].replace("PD", "").replace("GC", "").replace("MA", "").replace("NI", "")
            slice_info = parts[-1].replace(".npy", "").replace("slice", "Slice ")
            patients.setdefault(patient_id, {}).setdefault(region_id, []).append((file, slice_info))
    return patients

# -------------------------------------------------------
# Sort patients, regions, and slices numerically
# -------------------------------------------------------
def sort_patients(patients):
    """
    Sort the parsed patient dictionary:
    - Patients sorted by ID
    - Regions sorted by ID
    - Slices sorted by slice number
    """
    sorted_patients = {}
    for patient_id, regions in sorted(patients.items(), key=lambda x: int(x[0])):
        sorted_regions = {region_id: sorted(slices, key=lambda x: int(x[1].split()[-1]))
                          for region_id, slices in regions.items()}
        sorted_patients[patient_id] = sorted_regions
    return sorted_patients

# -------------------------------------------------------
# Export data as downloadable .npy or .png file
# -------------------------------------------------------
def export_file(data, file_type, file_name):
    """
    Export and offer data as a downloadable file.
    
    Args:
        data: Data to export (numpy array or image)
        file_type: "npy" for numpy array, "png" for image
        file_name: Name for the downloaded file (without extension)
    """
    if file_type == "npy":
        buffer = io.BytesIO()
        np.save(buffer, data)
        buffer.seek(0)
        st.download_button(
            label="Download as .npy",
            data=buffer,
            file_name=f"{file_name}.npy",
            mime="application/octet-stream",
        )
    elif file_type == "png":
        image = Image.fromarray(data)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
            label="Download as .png",
            data=buffer,
            file_name=f"{file_name}.png",
            mime="image/png",
        )
