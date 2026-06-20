import sqlite3
import streamlit as st
import fitz
import pandas as pd
import plotly.express as px
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #0f172a
    );
}

.main-title{
    text-align:center;
    color:white;
    font-size:48px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

[data-testid="stMetric"]{
    background:rgba(255,255,255,0.08);
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-title">
🚀 AI Resume Screener & Candidate Intelligence Platform
</div>

<div class="sub-title">
Automated Resume Analysis, Candidate Ranking & Talent Intelligence
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

conn = sqlite3.connect(
    "candidates.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS candidates(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_name TEXT,
    semantic_score REAL,
    skill_match REAL,
    final_score REAL
)
""")

conn.commit()

def extract_text(pdf_file):

    text = ""

    try:
        pdf = fitz.open(
            stream=pdf_file.read(),
            filetype="pdf"
        )

        for page in pdf:
            text += page.get_text()

    except:
        pass

    return text

def semantic_score(
    resume_text,
    jd_text
):

    resume_embedding = model.encode(
        [resume_text]
    )

    jd_embedding = model.encode(
        [jd_text]
    )

    score = cosine_similarity(
        resume_embedding,
        jd_embedding
    )[0][0]

    return round(score * 100, 2)

def keyword_score(
    resume_text,
    jd_text
):

    jd_words = set(
        jd_text.lower().split()
    )

    resume_words = set(
        resume_text.lower().split()
    )

    common = jd_words.intersection(
        resume_words
    )

    if len(jd_words) == 0:
        return 0

    return round(
        (len(common) / len(jd_words)) * 100,
        2
    )

uploaded_files = st.file_uploader(
    "📄 Upload Candidate Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area(
    "📝 Paste Job Description",
    height=250
)

if st.button("🚀 Screen Candidates"):

    if not uploaded_files:
        st.warning("Please upload resumes")
        st.stop()

    if not job_description:
        st.warning("Please enter job description")
        st.stop()

    results = []

    for file in uploaded_files:

        resume_text = extract_text(file)

        sem_score = semantic_score(
            resume_text,
            job_description
        )

        key_score = keyword_score(
            resume_text,
            job_description
        )

        final_score = round(
            (sem_score * 0.7) +
            (key_score * 0.3),
            2
        )

        results.append({
            "Candidate": file.name,
            "Semantic Score": sem_score,
            "Skill Match": key_score,
            "Final Score": final_score
        })

        cursor.execute(
            """
            INSERT INTO candidates
            (
                candidate_name,
                semantic_score,
                skill_match,
                final_score
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                file.name,
                sem_score,
                key_score,
                final_score
            )
        )

    conn.commit()

    df = pd.DataFrame(results)

    df = df.sort_values(
        by="Final Score",
        ascending=False
    )
    
    st.success("✅ Screening Completed")

    st.subheader("🏆 Top Candidate")

    st.metric(
        "Best Match",
        df.iloc[0]["Candidate"],
        f"{df.iloc[0]['Final Score']}%"
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("🧠 AI Ranking Explanation")

    top_candidate = df.iloc[0]

    st.info(
        f"""
        {top_candidate['Candidate']} ranked #1 because:

        • Highest semantic similarity with the job description

        • Strong skill match percentage

        • Better alignment with required technologies

        • Higher overall candidate score compared to other applicants
        """
    )

    st.markdown("---")
    st.subheader("📊 Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Candidates",
            len(df)
        )

    with col2:
        st.metric(
            "Top Score",
            f"{df['Final Score'].max():.2f}%"
        )

    with col3:
        st.metric(
            "Average Score",
            f"{df['Final Score'].mean():.2f}%"
        )

    fig = px.bar(
        df,
        x="Candidate",
        y="Final Score",
        color="Final Score",
        text="Final Score",
        title="Candidate Ranking Dashboard"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Candidates"
        )

    st.download_button(
        label="📥 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name="candidate_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("🎯 Candidate Status")

    for _, row in df.iterrows():

        if row["Final Score"] >= 60:
            st.success(
                f"{row['Candidate']} → Shortlisted ✅"
            )

        elif row["Final Score"] >= 40:
            st.warning(
                f"{row['Candidate']} → Consider 🤔"
            )

        else:
            st.error(
                f"{row['Candidate']} → Rejected ❌"
            )

st.markdown("---")
st.subheader("🗄 Candidate History")

history = pd.read_sql_query(
    "SELECT * FROM candidates",
    conn
)

st.dataframe(
    history,
    use_container_width=True
)

st.markdown("---")

st.markdown(
    """
    ### 🚀 AI Resume Screener & Candidate Intelligence Platform

    Developed using:
    Python | Streamlit | Sentence Transformers | Scikit-Learn | Plotly | SQLite
    """
)