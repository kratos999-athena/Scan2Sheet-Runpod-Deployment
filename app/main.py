import os
import streamlit as st
from PIL import Image
from app.pipeline import OCRPipeline

st.set_page_config(page_title="ScanToTable OCR", page_icon="🧊", layout="wide")

st.title("🧊 ScanToTable OCR Pipeline")
st.markdown("Upload your scanned document to extract tables automatically.")

DUMMY = os.environ.get("DUMMY_MODE", "0") == "1"
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "")

with st.sidebar:
    st.header("Settings")
    st.write("Dummy mode:", DUMMY)
    if st.button("Reload models"):
        OCRPipeline(dummy=DUMMY)._clear_cached_models()

uploaded_file = st.file_uploader("Upload PNG/JPG/PDF", type=['png', 'jpg', 'jpeg', 'pdf'])
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(uploaded_file.read())
        image_to_process = images[0] if images else None
    else:
        image_to_process = Image.open(uploaded_file).convert("RGB")

    if image_to_process:
        st.image(image_to_process, caption="Uploaded Image", width=400)
        if st.button("🚀 Start Table Extraction"):
            pipeline = OCRPipeline(dummy=DUMMY, model_dir=MODEL_DIR, hf_model_id=HF_MODEL_ID)

            with st.spinner("Preprocessing..."):
                preprocessed_cv = pipeline.preprocess_image_pil(image_to_process)
                st.image(Image.fromarray(preprocessed_cv[:, :, ::-1]), caption="Preprocessed")

            with st.spinner("Extracting table structure..."):
                extractor = pipeline.extract_table_structure(preprocessed_cv)
                st.image(Image.fromarray(extractor.combined_image), caption="Detected Table")

            with st.spinner("Running OCR..."):
                df = pipeline.run_ocr_and_build_table(extractor, preprocessed_cv)
                st.dataframe(df)
                st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "extracted_table.csv", "text/csv")
