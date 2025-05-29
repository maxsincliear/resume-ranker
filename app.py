# ========== IMPORTANT: PAGE CONFIG MUST BE FIRST ==========
import streamlit as st
st.set_page_config(
    page_title="Smart Resume Ranker", 
    layout="wide",
    page_icon="🧠"
)

# ========== REST OF IMPORTS ==========
import fitz  # PyMuPDF
import nltk
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ========== FIX FOR NLTK DATA ==========
@st.cache_resource
def load_nltk_data():
    # Create nltk_data directory in current working directory
    nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    
    # Add to NLTK data path
    nltk.data.path.append(nltk_data_path)
    
    # Download required NLTK data
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('wordnet', download_dir=nltk_data_path)
    nltk.download('omw-1.4', download_dir=nltk_data_path)

# Load NLTK data at startup
load_nltk_data()

# ========== PDF READING ==========
def read_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

# ========== TEXT PROCESSING ==========
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = nltk.word_tokenize(text)
        stop_words = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in stop_words]
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]
        return " ".join(lemmatized)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# ========== SIMILARITY CALCULATION ==========
def calculate_similarity(resume_text, job_desc):
    try:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
        return round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# ========== CUSTOM CSS ==========
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

# ========== MAIN APP ==========
st.markdown("<h1 style='text-align: center; color: white; background-color: #2a5298; padding: 1rem; border-radius: 10px;'>🧠 Smart Resume Ranker & Analysis</h1>", unsafe_allow_html=True)

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

st.subheader("Welcome to Smart Resume Ranker & Analysis 🚀")
st.markdown("Upload resumes, enter a job description, and get instant AI-based matching and analysis!")

uploaded_files = st.file_uploader("Upload one or more resumes", type="pdf", accept_multiple_files=True)
job_desc = st.text_area("Paste the job description here", height=200)

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
                    
                    st.write(f"**Match Score:** `{score}%`")
                    if score >= 70:
                        st.success("✅ Excellent match!")
                    elif score >= 50:
                        st.info("⚠️ Moderate match")
                    else:
                        st.warning("❌ Low match")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 14px;'>Project developed by <b>Vikas Jaipal</b> & <b>Sudha Koushal</b> — 3rd Year IT | Engineering College Bikaner</p>", unsafe_allow_html=True)
