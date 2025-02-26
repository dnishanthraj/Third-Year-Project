import streamlit as st
import os

# Configure the page
st.set_page_config(page_title="Welcome", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    /* Force Poppins on every element */
    [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
        font-family: 'Poppins', sans-serif !important;
    }
    .block-container {
        max-width: 1400px !important;  /* or 1600px, adjust to taste */
        padding: 3rem 2rem !important; /* Adjust as needed for your taste */
    }
                /* Increase vertical spacing between nav items */
    /* (Optional) If you want to reduce vertical margin inside sidebar expanders: */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        margin: 0.2rem 0 !important;
    }        
    # [data-testid="stSidebarNav"] ul {
    #     margin-top: 0.5rem; /* space above the list */
    # }
    [data-testid="stSidebarNav"] ul li {
        margin-bottom: 0.2rem; /* space between items */
    }
    /* Add some padding around each link */
    [data-testid="stSidebarNav"] ul li a {
        padding: 0.3rem 1rem; 
        border-radius: 4px;
    }
            /* Increase spacing before h2 headings in the sidebar */
    [data-testid="stSidebar"] h2 {
        margin-bottom: 1rem !important;
    }      
    </style>
    """, unsafe_allow_html=True)

st.title("Explainable AI Lung Nodule Segmentation Dashboard")
st.markdown("---")

st.markdown("""
Welcome to the Explainable AI Lung Nodule Segmentation Dashboard! Please select an option from the sidebar to continue.
""")