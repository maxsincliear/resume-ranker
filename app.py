import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required resources except punkt
nltk.download('stopwords')
nltk.download('wordnet')

# --- Text Preprocessing ---
def preprocess_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        # SAFE TOKENIZER - No punkt dependency
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)

        stop_words = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in stop_words]

        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(w) for w in tokens]

        return " ".join(lemmatized)
    except Exception as e:
        st.error(f"Text processing error: {str(e)}")
        return ""

# --- Extract Text from PDF ---
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# --- Calculate Similarity ---
def calculate_similarity(resume_text, job_desc):
    try:
        if not resume_text.strip() or not job_desc.strip():
            return 0.0
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([resume_text, job_desc])
        return round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)
    except Exception as e:
        st.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Resume Ranker", layout="wide")
st.title("🔍 Smart Resume Ranker and Analysis")
st.markdown("Developed by **Vikas Jaipal** & **Sudha Koushal**")

st.sidebar.title("📄 Upload Resume and Job Info")
resume_file = st.sidebar.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_title = st.sidebar.text_input("Job Title")
job_description = st.sidebar.text_area("Job Description")

if st.sidebar.button("Analyze"):
    if resume_file is None or job_description.strip() == "":
        st.warning("⚠️ Please upload a resume and enter a job description.")
    else:
        with st.spinner("Extracting and analyzing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            preprocessed_resume = preprocess_text(resume_text)
            preprocessed_jd = preprocess_text(job_description)
            match_score = calculate_similarity(preprocessed_resume, preprocessed_jd)

        st.subheader("📊 Match Score:")
        st.markdown(f"<h2 style='color: green;'>{match_score}%</h2>", unsafe_allow_html=True)

        if match_score >= 70:
            st.success("✅ Strong Match: Your resume aligns well with the job description!")
        elif match_score >= 40:
            st.info("ℹ️ Moderate Match: You might want to improve your resume.")
        else:
            st.error("❌ Low Match: Consider tailoring your resume for the job.")

        st.subheader("📄 Extracted Resume Text:")
        with st.expander("Click to view"):
            st.text(resume_text)

        st.subheader("📝 Job Description:")
        with st.expander("Click to view"):
            st.text(job_description)

