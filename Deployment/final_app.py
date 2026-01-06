import streamlit as st
from Ecg import ECG
import time

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .upload-text {
        font-size: 1.2rem;
        color: #262730;
        font-weight: 500;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://i.im.ge/2026/01/06/GodOkf.Gemini-Generated-Image-x04fsrx04fsrx04f-Photoroom.png", width=80)
    st.title("🫀 Silent Beat AI")
    st.markdown("---")
    
    st.markdown("### 🔍 How It Works")
    st.markdown("""
    1. **Upload** an ECG image
    2. **Processing** through AI pipeline
    3. **Analysis** of 13 ECG leads
    4. **Prediction** of cardiac condition
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Classification Types")
    st.success("✅ Normal ECG")
    st.error("⚠️ Myocardial Infarction")
    st.warning("⚠️ Abnormal Heartbeat")
    st.info("⚠️ History of MI")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI-powered system analyzes ECG images using:
    - Image Processing
    - Signal Extraction
    - Machine Learning
    - PCA Dimensionality Reduction
    """)

# Main header
st.title("❤️ SMI Disease Detection System")
st.markdown("### Advanced ECG Image Analysis using Computer Vision")
st.markdown("---")

# Initialize ECG object
ecg = ECG()

# File uploader with custom styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("### 📁 Upload your ECG image for analysis")
    uploaded_file = st.file_uploader(
        "Choose an ECG image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear ECG image for accurate analysis"
    )

if uploaded_file is not None:
    try:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Image Upload
        status_text.text("📤 Processing uploaded image...")
        progress_bar.progress(10)
        time.sleep(0.3)
        
        ecg_user_image_read = ecg.getImage(uploaded_file)
        
        st.markdown("---")
        st.markdown("## 🖼️ Original ECG Image")
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(ecg_user_image_read, caption="Uploaded ECG Image", use_column_width=True)
        
        # Step 2: Grayscale Conversion
        status_text.text("⚙️ Converting to grayscale...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
        
        with st.expander("🔲 View Grayscale Image", expanded=False):
            st.image(ecg_user_gray_image_read, caption="Grayscale Processed Image", use_column_width=True)
            st.caption("Grayscale conversion simplifies the image for better signal extraction")
        
        # Step 3: Dividing Leads
        status_text.text("✂️ Dividing ECG into 13 leads...")
        progress_bar.progress(35)
        time.sleep(0.3)
        
        dividing_leads = ecg.DividingLeads(ecg_user_image_read)
        
        with st.expander("📊 View Divided Leads (13 Leads)", expanded=False):
            st.markdown("**12 Standard Leads:**")
            st.image('Leads_1-12_figure.png', caption="Leads 1-12: Bipolar limb, Augmented, and Chest leads", use_column_width=True)
            st.markdown("**Long Lead 13:**")
            st.image('Long_Lead_13_figure.png', caption="Lead 13: Extended rhythm strip", use_column_width=True)
            st.caption("ECG divided into 13 standard medical leads for comprehensive analysis")
        
        # Step 4: Preprocessing
        status_text.text("🔧 Preprocessing leads (filtering & thresholding)...")
        progress_bar.progress(50)
        time.sleep(0.3)
        
        ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)
        
        with st.expander("🧹 View Preprocessed Leads", expanded=False):
            st.image('Preprossed_Leads_1-12_figure.png', caption="Preprocessed Leads 1-12", use_column_width=True)
            st.image('Preprossed_Leads_13_figure.png', caption="Preprocessed Lead 13", use_column_width=True)
            st.caption("Gaussian filtering and Otsu thresholding applied to enhance signal quality")
        
        # Step 5: Signal Extraction
        status_text.text("📈 Extracting signals using contour detection...")
        progress_bar.progress(65)
        time.sleep(0.3)
        
        ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
        
        with st.expander("📉 View Extracted Contour Signals", expanded=False):
            st.image('Contour_Leads_1-12_figure.png', caption="Contour-based Signal Extraction", use_column_width=True)
            st.caption("Contour detection isolates the ECG waveforms from background noise")
        
        # Step 6: 1D Conversion
        status_text.text("🔢 Converting to 1D signal array...")
        progress_bar.progress(75)
        time.sleep(0.3)
        
        ecg_1dsignal = ecg.CombineConvert1Dsignal()
        
        with st.expander("📊 View 1D Signal Data", expanded=False):
            st.dataframe(ecg_1dsignal)
            st.caption(f"Combined 1D signal with {ecg_1dsignal.shape[1]} features from all leads")
        
        # Step 7: Dimensionality Reduction
        status_text.text("🧮 Applying PCA dimensionality reduction...")
        progress_bar.progress(85)
        time.sleep(0.3)
        
        ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
        
        with st.expander("📐 View PCA Reduced Data", expanded=False):
            st.dataframe(ecg_final)
            st.caption(f"PCA reduced features: {ecg_final.shape[1]} principal components")
        
        # Step 8: Prediction
        status_text.text("🤖 Running ML model prediction...")
        progress_bar.progress(95)
        time.sleep(0.3)
        
        ecg_model = ecg.ModelLoad_predict(ecg_final)
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Display Result
        st.markdown("---")
        st.markdown("## 🎯 Diagnosis Result")
        
        # Create colored boxes based on prediction using native Streamlit components
        if "Normal" in ecg_model:
            st.success(f"### ✅ {ecg_model}")
            st.write("The ECG analysis indicates normal cardiac activity with no abnormalities detected.")
        elif "Myocardial Infarction" in ecg_model and "History" not in ecg_model:
            st.error(f"### ⚠️ {ecg_model}")
            st.write("The ECG shows patterns consistent with acute myocardial infarction. Immediate medical attention is recommended.")
        elif "Abnormal Heartbeat" in ecg_model:
            st.warning(f"### ⚠️ {ecg_model}")
            st.write("The ECG indicates irregular heartbeat patterns. Please consult a cardiologist for further evaluation.")
        else:
            st.info(f"### ℹ️ {ecg_model}")
            st.write("The ECG shows evidence of previous myocardial infarction. Regular monitoring is advised.")
        
        # Additional information
        st.markdown("---")
        st.info("⚕️ **Disclaimer:** This is an AI-based analysis tool. Always consult with a qualified healthcare professional for medical diagnosis and treatment.")
        
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")
        st.warning("Please ensure you've uploaded a valid ECG image and try again.")

else:
    # Instructions when no file is uploaded
    st.markdown("---")
    st.info("""
    ### 📝 Instructions:
    1. Click the **Browse files** button above
    2. Select an ECG image from your device
    3. Wait for the AI to process and analyze
    4. View the complete analysis pipeline
    5. Get your diagnosis result
    
    **Supported formats:** PNG, JPG, JPEG
    """)
    
    # Show example or demo section
    st.markdown("---")
    st.markdown("### 🎬 Features")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("#### 🖼️ Image Processing")
        st.write("Advanced image preprocessing and lead extraction")
    
    with col2:
        st.markdown("#### 📊 Signal Analysis")
        st.write("Contour-based 1D signal conversion")
    
    with col3:
        st.markdown("#### 🧮 PCA Reduction")
        st.write("Dimensionality reduction for optimal features")
    
    with col4:
        st.markdown("#### 🤖 ML Prediction")
        st.write("Pre-trained model for accurate diagnosis")
