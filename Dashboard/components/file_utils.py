# components/file_utils.py
import os
import streamlit as st
import numpy as np
from PIL import Image
import io

def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for the correct .npy file within the patient-specific subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    return file_path if os.path.exists(file_path) else None

def parse_filenames(files, prefix):
    """Parse filenames to group by Patient ID, Region ID, and Slices."""
    patients = {}
    for file in files:
        if prefix in file:
            parts = file.split("_")
            patient_id = parts[0]
            region_id = parts[1].replace("PD", "").replace("GC", "").replace("MA", "").replace("NI", "")
            slice_info = parts[-1].replace(".npy", "").replace("slice", "Slice ")
            patients.setdefault(patient_id, {}).setdefault(region_id, []).append((file, slice_info))
    return patients

def sort_patients(patients):
    """Sort patients, regions, and slices numerically."""
    sorted_patients = {}
    for patient_id, regions in sorted(patients.items(), key=lambda x: int(x[0])):
        sorted_regions = {region_id: sorted(slices, key=lambda x: int(x[1].split()[-1]))
                          for region_id, slices in regions.items()}
        sorted_patients[patient_id] = sorted_regions
    return sorted_patients

def export_file(data, file_type, file_name):
    """Export data as a file for download."""
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