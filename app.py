import streamlit as st
import fitz  # PyMuPDF
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ========== NLTK DATA DOWNLOAD (FIXES ERROR) ==========
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ========== PDF READING FUNCTION ==========
def read_pdf(file):
    """Reads PDF content using PyMuPDF"""
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

# ========== TEXT PREPROCESSING ==========
def preprocess_text(text):
    """Cleans and tokenizes text"""
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase and remove special chars
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize and remove stopwords
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(lemmatized)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# ========== SIMILARITY CALCULATION ==========
def calculate_similarity(resume_text, job_desc):
    """Calculates match percentage using TF-IDF"""
    try:
        if not resume_text or not job_desc:
            return 0.0
            
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(similarity * 100, 2)
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# ========== STREAMLIT UI ==========
st.set_page_config(
    page_title="Smart Resume Ranker", 
    layout="wide",
    page_icon="🧠"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #2a5298;
        color: white;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        min-height: 200px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: white; background-color: #2a5298; padding: 1rem; border-radius: 10px;'>🧠 Smart Resume Ranker & Analysis</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3178/3178374.png", width=80)
    st.markdown("## 🔍 Navigation")
    st.markdown("✅ Upload Resume(s)")
    st.markdown("✅ Enter Job Description")
    st.markdown("✅ View Matching Results")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developed by:")
    st.markdown("**Vikas Jaipal (22EEBIT009)**")
    st.markdown("**Sudha Koushal (22EEBIT006)**")

# Main Content
st.subheader("Welcome to Smart Resume Ranker & Analysis 🚀")
st.markdown("Upload resumes, enter a job description, and get instant AI-based matching and analysis!")

# File Upload
st.markdown("### 📂 Step 1: Upload Resume PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more resumes", 
    type="pdf", 
    accept_multiple_files=True,
    help="Upload PDF resumes only"
)

# Job Description
st.markdown("### 📝 Step 2: Enter Job Description")
job_desc = st.text_area(
    "Paste the job description here", 
    height=200,
    placeholder="Paste job description text here..."
)

# Analysis Button
if st.button("🔍 Analyze & Match", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one resume.")
    elif not job_desc.strip():
        st.warning("Please enter a job description.")
    else:
        with st.spinner("Analyzing resumes... Please wait ⏳"):
            job_clean = preprocess_text(job_desc)
            
            for file in uploaded_files:
                with st.expander(f"📄 {file.name}", expanded=True):
                    text = read_pdf(file)
                    if not text:
                        st.error("Could not read PDF content")
                        continue
                        
                    resume_clean = preprocess_text(text)
                    score = calculate_similarity(resume_clean, job_clean)
                    
                    # Display results
                    st.write(f"**Match Score:** `{score}%`")
                    if score >= 70:
                        st.success("✅ Excellent match!")
                    elif score >= 50:
                        st.info("⚠️ Moderate match")
                    else:
                        st.warning("❌ Low match")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Project developed by <b>Vikas Jaipal</b> & <b>Sudha Koushal</b> — 3rd Year IT | Engineering College Bikaner</p>", unsafe_allow_html=True)
