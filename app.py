import streamlit as st
import torch
import json
import spacy
import os
import numpy as np
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from sentence_transformers import SentenceTransformer, util

st.set_page_config(
    page_title="Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

CLF_PATH = r"resume_classifier"

@st.cache_resource
def load_all_models():
    with open(os.path.join(CLF_PATH, "id2label.json")) as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(CLF_PATH, "label_map.json")) as f:
        label_map = json.load(f)
    with open(os.path.join(CLF_PATH, "metrics.json")) as f:
        metrics = json.load(f)

    tokenizer_clf = DistilBertTokenizer.from_pretrained(CLF_PATH)
    model_clf = DistilBertForSequenceClassification.from_pretrained(
        CLF_PATH, low_cpu_mem_usage=True)
    model_clf.eval()

    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model     = T5ForConditionalGeneration.from_pretrained("t5-small")
    t5_model.eval()

    nlp = spacy.load("en_core_web_sm")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    return (model_clf, tokenizer_clf, id2label,
            label_map, metrics, t5_model, t5_tokenizer, nlp, sbert)

with st.spinner("Loading models — first load takes 2-3 mins..."):
    (model_clf, tokenizer_clf, id2label,
     label_map, metrics, t5_model, t5_tokenizer, nlp, sbert) = load_all_models()

device = torch.device("cpu")
model_clf.to(device)
t5_model.to(device)

def classify_resume(text):
    inputs = tokenizer_clf(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        logits = model_clf(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0].numpy()
    pred  = int(probs.argmax())
    return id2label[pred], float(probs[pred]) * 100, probs

def summarize_resume(text):
    try:
        input_text = "summarize: " + text[:600]
        inputs = t5_tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs["input_ids"],
                max_length=120,
                min_length=30,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Summary unavailable: {str(e)}"

def extract_entities(text):
    doc = nlp(text[:5000])
    entities = {"ORG": [], "GPE": [], "DATE": [], "SKILLS": []}

    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text.split()) > 1:
            if ent.text not in entities["ORG"]:
                entities["ORG"].append(ent.text)
        elif ent.label_ == "GPE":
            if ent.text not in entities["GPE"]:
                entities["GPE"].append(ent.text)
        elif ent.label_ == "DATE":
            if ent.text not in entities["DATE"]:
                entities["DATE"].append(ent.text)

    skills = [
        "Python","Java","JavaScript","SQL","Machine Learning",
        "Deep Learning","NLP","React","TensorFlow","PyTorch",
        "AWS","Docker","Git","C++","R","Tableau","Power BI",
        "MongoDB","Django","Flask","Pandas","NumPy","Scikit-learn",
        "HR","Human Resources","Recruitment","Payroll","HRIS",
        "Excel","SAP","Communication","Leadership","Project Management",
        "Agile","Scrum","DevOps","Kubernetes","Linux","Azure","GCP",
        "Hadoop","Spark","Kafka","Spring","Hibernate","Node.js"
    ]
    for s in skills:
        if s.lower() in text.lower() and s not in entities["SKILLS"]:
            entities["SKILLS"].append(s)

    return entities

def job_match(resume_text, job_text):
    r_emb = sbert.encode(resume_text[:1000], convert_to_tensor=True)
    j_emb = sbert.encode(job_text, convert_to_tensor=True)
    score = float(util.cos_sim(r_emb, j_emb)[0][0]) * 100
    level = (
        "Excellent match" if score >= 75 else
        "Good match"      if score >= 55 else
        "Partial match"   if score >= 35 else
        "Low match"
    )
    return round(score, 1), level

st.title("Resume Analyzer")
st.caption("DistilBERT classification  |  T5 summarization  |  spaCy NER  |  SBERT job matching")
st.divider()

tab1, tab2, tab3 = st.tabs([
    "Resume Analyzer",
    "Job Match",
    "Model Performance"
])

with tab1:
    st.subheader("Paste your resume")

    example = """John Smith | Python Developer | john@email.com | Mumbai

EXPERIENCE
Senior Python Developer at Google India (2021-2024)
Built ML pipelines using TensorFlow and PyTorch
Developed REST APIs with Django and Flask
Worked with AWS, Docker and Kubernetes for deployment

Data Scientist at TCS (2019-2021)
Applied Machine Learning and NLP on customer data
Used Pandas, NumPy, Scikit-learn for data analysis
SQL database management with PostgreSQL and MongoDB

EDUCATION
B.Tech Computer Science - IIT Bombay (2019)

SKILLS: Python, Machine Learning, Deep Learning, NLP, SQL,
TensorFlow, PyTorch, AWS, Docker, Git, Django, Flask, Pandas"""

    resume_text = st.text_area(
        "Resume text:",
        value=example,
        height=280,
        placeholder="Paste full resume here..."
    )

    if st.button("Analyze Resume", type="primary"):
        if resume_text.strip():
            with st.spinner("Running 4 models — please wait..."):
                category, confidence, all_probs = classify_resume(resume_text)
                summary  = summarize_resume(resume_text)
                entities = extract_entities(resume_text)

            st.divider()

            st.subheader("Professional Summary (T5)")
            st.info(summary)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Predicted Category (DistilBERT)")
                st.success(f"**{category}**")
                st.metric("Confidence", f"{confidence:.1f}%")
                color = (
                    "green"  if confidence >= 70 else
                    "orange" if confidence >= 40 else
                    "red"
                )
                st.markdown(
                    f"<div style='background:{color};height:10px;"
                    f"border-radius:5px;width:{min(confidence,100):.0f}%'>"
                    f"</div>", unsafe_allow_html=True
                )

            with col2:
                st.subheader("Top 5 Predictions")
                top5 = np.argsort(all_probs)[::-1][:5]
                for idx in top5:
                    lbl  = id2label[idx]
                    prob = float(all_probs[idx])
                    st.write(f"`{lbl}` — {prob*100:.1f}%")
                    st.progress(prob)

            st.divider()

            st.subheader("Extracted Information (spaCy NER)")
            col3, col4, col5, col6 = st.columns(4)

            with col3:
                st.markdown("**Skills found**")
                if entities["SKILLS"]:
                    for s in entities["SKILLS"]:
                        st.markdown(
                            f"<span style='background:#E6F1FB;"
                            f"color:#0C447C;padding:2px 8px;"
                            f"border-radius:10px;font-size:12px;"
                            f"margin:2px;display:inline-block'>"
                            f"{s}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("None found")

            with col4:
                st.markdown("**Organisations**")
                if entities["ORG"]:
                    for o in entities["ORG"][:6]:
                        st.markdown(f"- {o}")
                else:
                    st.caption("None found")

            with col5:
                st.markdown("**Locations**")
                if entities["GPE"]:
                    for g in entities["GPE"][:6]:
                        st.markdown(f"- {g}")
                else:
                    st.caption("None found")

            with col6:
                st.markdown("**Dates / Experience**")
                if entities["DATE"]:
                    for d in entities["DATE"][:6]:
                        st.markdown(f"- {d}")
                else:
                    st.caption("None found")
        else:
            st.warning("Please paste a resume first!")

with tab2:
    st.subheader("Match Resume to Job Description")
    st.caption("Paste resume and job description — get instant similarity score")

    col_l, col_r = st.columns(2)
    with col_l:
        resume_match = st.text_area(
            "Resume:", height=260,
            placeholder="Paste resume here..."
        )
    with col_r:
        job_desc = st.text_area(
            "Job Description:", height=260,
            placeholder="Paste job description here..."
        )

    if st.button("Calculate Match Score", type="primary"):
        if resume_match.strip() and job_desc.strip():
            with st.spinner("Computing similarity..."):
                score, level = job_match(resume_match, job_desc)

            st.divider()

            col1, col2, col3 = st.columns(3)
            col1.metric("Match Score", f"{score}%")
            col2.metric("Match Level", level)
            col3.metric("Recommendation",
                        "Apply!" if score >= 55 else "Tailor resume first")

            color = (
                "#4CAF50" if score >= 75 else
                "#2196F3" if score >= 55 else
                "#FF9800" if score >= 35 else
                "#F44336"
            )
            st.markdown(
                f"<div style='background:{color};border-radius:12px;"
                f"padding:24px;text-align:center;color:white;margin:16px 0'>"
                f"<div style='font-size:48px;font-weight:500'>{score}%</div>"
                f"<div style='font-size:18px;margin-top:8px'>{level}</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.progress(score / 100)

            if score >= 75:
                st.success("Strong match — resume aligns well with this role!")
            elif score >= 55:
                st.info("Good match — candidate meets most requirements.")
            elif score >= 35:
                st.warning("Partial match — consider adding relevant keywords.")
            else:
                st.error("Low match — this role may not suit your background.")

            st.divider()
            st.subheader("Tips to improve your match score")
            st.markdown("""
            - Add keywords directly from the job description to your resume
            - Quantify achievements with numbers and percentages
            - Mirror the job title in your resume headline
            - Include required tools and technologies explicitly
            """)
        else:
            st.warning("Please paste both resume and job description!")

with tab3:
    st.subheader("DistilBERT Classifier Performance")

    acc    = metrics["accuracy"]
    report = metrics["classification_report"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",   f"{acc*100:.2f}%", delta="target: 85%")
    col2.metric("Macro F1",        f"{report['macro avg']['f1-score']*100:.2f}%")
    col3.metric("Macro Precision", f"{report['macro avg']['precision']*100:.2f}%")
    col4.metric("Macro Recall",    f"{report['macro avg']['recall']*100:.2f}%")

    st.divider()
    st.markdown("**Per-Class Performance**")

    class_data = {
        "Category":  [],
        "Precision": [],
        "Recall":    [],
        "F1 Score":  [],
        "Samples":   []
    }
    for lbl in metrics["label_names"]:
        if lbl in report:
            class_data["Category"].append(lbl)
            class_data["Precision"].append(
                f"{report[lbl]['precision']*100:.1f}%")
            class_data["Recall"].append(
                f"{report[lbl]['recall']*100:.1f}%")
            class_data["F1 Score"].append(
                f"{report[lbl]['f1-score']*100:.1f}%")
            class_data["Samples"].append(int(report[lbl]['support']))

    st.dataframe(class_data, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Models Used in This Project")
    st.markdown("""
    | Model | Task | Architecture | Parameters |
    |-------|------|-------------|-----------|
    | DistilBERT | Resume category classification | Encoder-only transformer | 66M |
    | T5-small | Resume summarization | Encoder-Decoder transformer | 60M |
    | spaCy en_core_web_sm | Named entity recognition | CNN + rule-based | 12M |
    | SBERT all-MiniLM-L6-v2 | Job match similarity | Siamese BERT | 22M |

    **Training details:**
    - Optimizer: AdamW with weight decay 0.01
    - Learning rate: 2e-5 with 10% warmup scheduler
    - Early stopping: patience = 2 epochs
    - fp16: disabled (avoids gradient underflow)
    - Hardware: Google Colab T4 GPU
    """)