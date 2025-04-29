# -------------------------------------------------------
# Contact & Support Page - Explainable AI Lung Nodule Segmentation
# -------------------------------------------------------

import streamlit as st
import smtplib
import os

# -------------------------------------------------------
# Email Configuration
# -------------------------------------------------------
# Set your support email address here
EMAIL_ADDRESS = "dnishanthraj@gmail.com"  # Replace with your email
EMAIL_PASSWORD = "..."  # Omitted for security reasons.

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(page_title="Support & FAQ", layout="wide")

# -------------------------------------------------------
# Custom CSS Styling for fonts and layout
# -------------------------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
    /* Force Poppins font on all elements */
    [data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
        font-family: 'Poppins', sans-serif !important;
    }
    /* Adjust block container width and padding */
    .block-container {
        max-width: 1400px !important;
        padding: 3rem 2rem !important;
    }
    /* Increase vertical spacing between sidebar expanders */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        margin: 0.2rem 0 !important;
    }
    /* Adjust sidebar navigation item spacing */
    [data-testid="stSidebarNav"] ul li {
        margin-bottom: 0.2rem;
    }
    /* Add padding around sidebar links */
    [data-testid="stSidebarNav"] ul li a {
        padding: 0.3rem 1rem; 
        border-radius: 4px;
    }
    /* Increase spacing before h2 headings in sidebar */
    [data-testid="stSidebar"] h2 {
        margin-bottom: 1rem !important;
    }      
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Page Title and Introduction
# -------------------------------------------------------
st.title("Contact & Support")
st.markdown("---")

st.markdown("""
This page provides answers to frequently asked questions about the Lung Nodule Segmentation Dashboard and offers contact information for further assistance.
""")

# -------------------------------------------------------
# FAQ Section
# -------------------------------------------------------
st.header("Frequently Asked Questions")

with st.expander("What is the purpose of this dashboard?"):
    st.markdown("""
The dashboard is designed to help radiologists review automated lung nodule segmentation results. It provides both an overview of model performance through metrics and the ability to inspect individual scan slices visually.
    """)

with st.expander("How are the segmentation results evaluated?"):
    st.markdown("""
The segmentation performance is measured using metrics such as **Dice Score** and **Intersection over Union (IoU)**, along with detection metrics like **Precision**, **Recall**, and **FPPS (False Positives per Scan)**. These metrics help quantify the accuracy and reliability of the segmentation.
    """)

with st.expander("What is False Positive Reduction (FPR)?"):
    st.markdown("""
FPR is a post-processing step that uses an additional classifier (implemented with XGBoost) to filter out spurious detections from the segmentation output. This step improves the overall accuracy by reducing the number of false positives.
    """)

with st.expander("How do I navigate the dashboard?"):
    st.markdown("""
- **Sidebar Navigation:** Use the sidebar to select the model folder, toggle between raw and FPR results, and choose between "No Clean" and "Clean" data.
- **Pages:** The dashboard is divided into several pages:
  - **Introduction:** Provides an overview of the project and dashboard.
  - **Guide:** Offers detailed explanations of metrics and functionality.
  - **Slice Results:** Allows you to inspect individual scan slices with various overlays.
  - **Support & FAQ:** (this page) Provides help and contact information.
    """)

# -------------------------------------------------------
# Contact Form Section
# -------------------------------------------------------
st.header("Contact Support")

st.markdown("""
If you have any questions or need further assistance, please submit your query using the form below.
""")

with st.form(key="contact_form"):
    # Input fields for user information
    user_name = st.text_input("Your Name")
    user_email = st.text_input("Your Email", help="Provide a valid email so we can respond.")
    user_message = st.text_area("Your Message", help="Describe your issue or question in detail.")
    
    # Submit button for the form
    submit_button = st.form_submit_button("Send Message")

    if submit_button:
        if user_name and user_email and user_message:
            try:
                # -------------------------------------------------------
                # Email Sending Logic
                # -------------------------------------------------------
                with smtplib.SMTP("smtp.gmail.com", 587) as server:
                    server.starttls()
                    server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                    subject = f"Dashboard Query from {user_name}"
                    body = f"Name: {user_name}\nEmail: {user_email}\n\nMessage:\n{user_message}"
                    email_message = f"Subject: {subject}\n\n{body}"
                    server.sendmail(user_email, EMAIL_ADDRESS, email_message)
                
                st.success("Your message has been sent successfully. We will get back to you shortly.")
            except Exception as e:
                st.error(f"Failed to send message. Error: {e}")
        else:
            st.warning("Please fill out all fields before submitting.")

# -------------------------------------------------------
# Quick Tips Section
# -------------------------------------------------------
st.header("Quick Tips")
st.markdown("""
- **Start with the Introduction:** Get a high-level overview of the project.
- **Refer to the Guide:** Understand the key metrics and tools.
- **Check the FAQ:** Find answers to common questions.
- **Contact Us:** Reach out if you need additional support.
""")

# -------------------------------------------------------
# End of Contact & Support Page
# -------------------------------------------------------
st.markdown("---")
