# ats_score_checker.py

# 🛠️ First install required libraries:
# pip install streamlit sentence-transformers PyMuPDF python-docx

import streamlit as st

st.set_page_config(page_title="ATS Resume Score Checker", layout="centered")

from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import docx  # python-docx

# Load the sentence transformer model
@st.cache_resource
def load_model():
    with st.spinner("Loading model... (this may take up to a minute the first time)"):
        return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# 📘 Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        st.stop()
    if not text.strip():
        st.error("❌ Could not extract text from the uploaded PDF. Make sure your PDF contains selectable text (not just images).")
        st.stop()
    return text

# 📘 Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    text = ""
    try:
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        st.stop()
    if not text.strip():
        st.error("❌ Could not extract text from the uploaded DOCX. Make sure your DOCX contains readable text.")
        st.stop()
    return text

# 🧠 Function to calculate ATS score
def calculate_ats_score(resume_text, job_desc):
    try:
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        job_embedding = model.encode(job_desc, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
        return round(similarity_score * 100, 2)
    except Exception as e:
        st.error(f"Error calculating ATS score: {e}")
        return 0.0

# --- SIDEBAR ---
st.sidebar.title("ATS Resume Score Checker")
st.sidebar.info(
    "Upload your resume (PDF/DOCX) or paste the text. "
    "Paste the job description. Click 'Check ATS Score' to see how well your resume matches!"
)
st.sidebar.markdown("---")
upload_option = st.sidebar.radio("Resume Input Method", ("Upload PDF/DOCX", "Paste Text"))

# --- MAIN PAGE ---
st.title("📄 ATS Resume Score Checker")
st.write("Check how well your resume matches a job description (simulated ATS).")

# Limit text length for faster processing
MAX_CHARS = 2000

resume_text = ""
if upload_option == "Upload PDF/DOCX":
    uploaded_resume = st.sidebar.file_uploader("Upload your Resume (PDF or DOCX only)", type=["pdf", "docx"])
    if uploaded_resume:
        if uploaded_resume.name.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_resume)
        elif uploaded_resume.name.lower().endswith(".docx"):
            resume_text = extract_text_from_docx(uploaded_resume)
        else:
            st.sidebar.error("Unsupported file type.")
        if resume_text.strip():
            st.success("✅ Resume text extracted successfully.")
            with st.expander("🔍 View Extracted Resume Text"):
                st.text_area("", resume_text, height=300)
else:
    resume_text = st.text_area("✍️ Paste your Resume Text", height=300)

# Job description input
job_desc = st.text_area("📝 Paste the Job Description", height=300)

# Limit input length for speed
resume_text = resume_text[:MAX_CHARS]
job_desc = job_desc[:MAX_CHARS]

# Check ATS Score
if st.button("🔎 Check ATS Score"):
    if resume_text.strip() and job_desc.strip():
        with st.spinner("Calculating ATS score..."):
            score = calculate_ats_score(resume_text, job_desc)
        st.subheader(f"📊 Your ATS Compatibility Score: {score}%")

        # Feedback messages
        if score >= 75:
            st.success("✅ Great! Your resume aligns very well with the job description.")
        elif score >= 50:
            st.warning("⚠️ Average match. Consider improving keywords and role-specific content.")
        else:
            st.error("❌ Low match. Consider rewriting your resume to better fit the job description.")

        st.caption("Note: This is an approximate semantic match, simulating basic ATS logic.")
    else:
        st.warning("⚠️ Please provide both resume and job description text to calculate the ATS score.")

# --- FOOTER ---
# Add this at the very end of your file

st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 10px;">
    <div style="text-align: center; color: gray; font-size: 0.9em;">
        © 2025 ResumeBuilder. All rights reserved. Lokesh Boligorla.
    </div>
    """,
    unsafe_allow_html=True
)