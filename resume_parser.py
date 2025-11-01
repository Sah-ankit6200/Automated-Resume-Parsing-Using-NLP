import fitz  # PyMuPDF for PDF parsing
import docx  # python-docx for DOCX parsing
import re
import nltk
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Stopwords for filtering
STOP_WORDS = set(stopwords.words('english'))

def extract_email(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    return email[0] if email else 'N/A'

def extract_phone(text):
    phone = re.findall(r'\+?\d[\d -]{8,}\d', text)
    return phone[0] if phone else 'N/A'

def extract_name(text):
    """Extracts the candidate's name from the resume."""
    lines = text.strip().split("\n")

    # Check the first 15 lines for a probable name
    for line in lines[:15]:  
        line = line.strip()
        
        # Match common name formats (First Last, First Middle Last)
        if re.match(r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+$", line):
            return line.strip()

        # Check for labels like "Name:" or "Full Name:"
        match = re.search(r"(Name|Full Name)[:\s]+(.+)", line, re.IGNORECASE)
        if match:
            return match.group(2).strip()

    return "N/A"

def extract_education(text):
    education_keywords = ["bachelor", "master", "ph.d", "university", "college", "degree"]
    education = [line.strip() for line in text.split('\n') if any(word in line.lower() for word in education_keywords)]
    return ', '.join(education) if education else 'N/A'

def extract_skills(text):
    skills_section = re.search(r'(Skills|Technical Skills|Key Skills)([\s\S]*?)(Experience|Projects|Education|$)', text, re.IGNORECASE)
    if skills_section:
        skills_text = skills_section.group(2)
        skills = re.split(r'[\n,;]', skills_text)
        skills = [skill.strip() for skill in skills if skill.strip() and skill.strip().lower() not in STOP_WORDS]
        return set(skills)
    return set()

def extract_projects(text):
    projects_section = re.search(r'(Projects|Project Work|Key Projects|Professional Projects)([\s\S]*?)(Experience|Education|Skills|Certifications|$)', text, re.IGNORECASE)
    if projects_section:
        projects_text = projects_section.group(2)
        projects = re.split(r'\n|\t|•|-', projects_text)
        return ', '.join([proj.strip() for proj in projects if proj.strip()])
    return 'N/A'

def extract_experience(text):
    experience_section = re.search(r'(Experience|Work History|Employment)([\s\S]*?)(Education|Skills|Certifications|$)', text, re.IGNORECASE)
    if experience_section:
        experience_text = experience_section.group(2)
        experience = re.split(r'\n|\t|•|-', experience_text)
        return ', '.join([exp.strip() for exp in experience if exp.strip()])
    return 'N/A'

def parse_resume(file, file_type):
    text = ""
    
    if file_type == "pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    elif file_type == "docx":
        doc = docx.Document(file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    
    # Debugging: Print the first 20 lines of the extracted text
    print("\n----- Extracted Text Preview -----\n")
    print("\n".join(text.split("\n")[:20]))  # Show only the first 20 lines for debugging
    print("\n-----------------------------------\n")
    
    return {
        'name': extract_name(text),
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'projects': extract_projects(text),
        'experience': extract_experience(text)
    }

def find_non_matching_skills(resume_skills, job_description):
    """
    Compares resume skills with job description and finds the missing skills.

    :param resume_skills: List of skills extracted from the resume
    :param job_description: Job description text
    :return: List of missing skills
    """
    if not resume_skills or not job_description:
        return []

    job_skills = set(job_description.lower().split())  # Convert job description to a set of words
    missing_skills = [skill for skill in resume_skills if skill.lower() not in job_skills]

    return missing_skills

### **🔹 Text Preprocessing & Similarity Calculation**
def preprocess(text):
    """Clean and preprocess text"""
    words = re.sub(r'\W+', ' ', text).lower().split()
    return ' '.join([word for word in words if word not in STOP_WORDS])

def calculate_evaluation_metrics(resume_skills, job_skills):
    """Calculate precision, recall, and F1-score"""
    resume_skills_set = set(preprocess(', '.join(resume_skills)).split())
    job_skills_set = set(preprocess(job_skills).split())

    true_positives = len(resume_skills_set.intersection(job_skills_set))
    false_positives = len(resume_skills_set - job_skills_set)
    false_negatives = len(job_skills_set - resume_skills_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return round(precision, 2), round(recall, 2), round(f1, 2)

def calculate_similarity(resume_details, job_description):
    """Calculate multiple similarity scores including BEART and SBERT"""
    try:
        # Prepare texts
        resume_text = preprocess(resume_details['education'])
        skills_text = preprocess(', '.join(resume_details['skills']))
        job_text = preprocess(job_description)
        
        # Combine all resume text for SBERT
        full_resume_text = " ".join([
            resume_details['education'],
            " ".join(resume_details['skills']),
            resume_details.get('experience', ''),
            resume_details.get('projects', '')
        ])
        
        vectorizer = TfidfVectorizer()

        # 1. Education Similarity (TF-IDF)
        tfidf_matrix_education = vectorizer.fit_transform([resume_text, job_text])
        education_similarity = cosine_similarity(tfidf_matrix_education[0:1], tfidf_matrix_education[1:2])[0][0]

        # 2. Skills Similarity (TF-IDF)
        tfidf_matrix_skills = vectorizer.fit_transform([skills_text, job_text])
        skills_similarity = cosine_similarity(tfidf_matrix_skills[0:1], tfidf_matrix_skills[1:2])[0][0]

        # 3. BEART Similarity (Keyword-based)
        beart_similarity = calculate_beart_similarity(resume_details['skills'], job_description)

        # 4. SBERT Similarity (Semantic)
        sbert_score = calculate_sbert_similarity(full_resume_text, job_description)

        # Calculate overall similarity (weighted average)
        overall_similarity = (skills_similarity + education_similarity + 
                            (beart_similarity/100) + (sbert_score/100)) / 4

        precision, recall, f1 = calculate_evaluation_metrics(resume_details['skills'], job_description)

        return (
            round(skills_similarity * 100, 2),      # TF-IDF Skills Score
            round(education_similarity * 100, 2),   # TF-IDF Education Score
            round(beart_similarity, 2),             # BEART Score
            round(sbert_score, 2),                  # SBERT Score
            round(overall_similarity * 100, 2),     # Combined Overall Score
            precision,                              # Precision
            recall,                                 # Recall
            f1                                      # F1 Score
        )
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def calculate_sbert_similarity(resume_text, job_description):
    """Calculate semantic similarity using SBERT embeddings"""
    try:
        # Encode both texts
        resume_embedding = SBERT_MODEL.encode(resume_text, convert_to_tensor=True)
        jd_embedding = SBERT_MODEL.encode(job_description, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding)
        
        # Convert to percentage
        return round(similarity.item() * 100, 2)
    except Exception as e:
        print(f"Error calculating SBERT similarity: {e}")
        return 0.0
    
### **🔹 Job Description Scraping & ATS Score Calculation**
def get_job_description(url):
    """Scrape job description from URL"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    job_desc = soup.find(class_="jobDescriptionText")
    return job_desc.get_text() if job_desc else soup.get_text()

def calculate_ats_score(precision, recall, f1):
    """Calculate ATS score based on precision, recall, and F1-score"""
    return round(((precision + recall + f1) / 3) * 100, 2)


def calculate_beart_similarity(resume_skills, job_description):
    """
    Calculate BEART Similarity score between resume skills and job description
    BEART = Basic Exact And Related Term similarity
    """
    if not resume_skills or not job_description:
        return 0.0
    
    # Preprocess both sets
    resume_skills = [skill.lower().strip() for skill in resume_skills]
    job_desc_skills = set(re.findall(r'\b[\w-]+\b', job_description.lower()))
    
    # Calculate exact matches
    exact_matches = sum(1 for skill in resume_skills if skill in job_desc_skills)
    
    # Calculate related terms (partial matches)
    related_matches = 0
    for skill in resume_skills:
        for jd_skill in job_desc_skills:
            if skill in jd_skill or jd_skill in skill:
                related_matches += 0.5  # Partial match score
                break
    
    total_possible = len(resume_skills)
    if total_possible == 0:
        return 0.0
    
    # BEART score formula: (exact_matches + related_matches) / total_skills
    beart_score = (exact_matches + related_matches) / total_possible
    
    return round(beart_score, 2)  # Convert to percentage


# Load SBERT model (add at top of file)
SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model good for this use case

def calculate_sbert_similarity(resume_text, job_description):
    """
    Calculate semantic similarity using SBERT embeddings
    """
    try:
        # Encode both texts
        resume_embedding = SBERT_MODEL.encode(resume_text, convert_to_tensor=True)
        jd_embedding = SBERT_MODEL.encode(job_description, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding)
        
        # Convert to percentage
        return round(similarity.item() , 2)
    except Exception as e:
        print(f"Error calculating SBERT similarity: {e}")
        return 0.0