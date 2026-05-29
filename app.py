import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

st.set_page_config(page_title="AI Resume Screening System")

st.title("AI Resume Screening & Candidate Ranking System")

job_description = st.text_area(
    "Paste Job Description"
)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def extract_text(pdf_file):

    text = ""

    reader = PyPDF2.PdfReader(pdf_file)

    for page in reader.pages:

        if page.extract_text():

            text += page.extract_text()

    return text


if st.button("Rank Candidates"):

    if job_description and uploaded_files:

        jd_embedding = model.encode(
            [job_description]
        )

        results = []

        for resume in uploaded_files:

            resume_text = extract_text(
                resume
            )

            resume_embedding = model.encode(
                [resume_text]
            )

            score = cosine_similarity(
                jd_embedding,
                resume_embedding
            )[0][0]

            results.append(
                {
                    "Candidate": resume.name,
                    "Match Score (%)":
                    round(score * 100, 2)
                }
            )

        df = pd.DataFrame(results)

        df = df.sort_values(
            by="Match Score (%)",
            ascending=False
        )

        st.success(
            "Ranking Completed"
        )

        st.dataframe(df)

    else:

        st.warning(
            "Upload resumes and job description."
        )
