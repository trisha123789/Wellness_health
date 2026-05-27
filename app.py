


import streamlit as st
import pandas as pd
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #0f172a, #111827);
    color: white;
}

.title {
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:30px;
}

.stTextArea textarea {
    background-color:#1e293b !important;
    color:white !important;
    border-radius:15px !important;
    border:1px solid #475569 !important;
    font-size:18px !important;
}

.result-box {
    background:#1e293b;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 0px 15px rgba(0,0,0,0.5);
}

.tip-box {
    background:#111827;
    padding:18px;
    border-radius:16px;
    margin-top:15px;
    border-left:5px solid #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# DATASET
# =========================================================

data = {
    "text": [

        # NORMAL
        "I feel happy today",
        "Life is going well",
        "I am enjoying my studies",
        "I feel peaceful and calm",

        # STRESS
        "I am mentally exhausted",
        "Too much pressure on me",
        "I cannot manage my workload",
        "Everything feels overwhelming",

        # ANXIETY
        "I keep worrying about everything",
        "I feel nervous all the time",
        "I am scared about my future",
        "I overthink every situation",

        # DEPRESSION
        "I feel empty inside",
        "Nothing makes me happy anymore",
        "I feel lonely and hopeless",
        "I lost interest in life",

        # SUICIDAL
        "I want to disappear forever",
        "Nobody would care if I die",
        "I cannot continue anymore",
        "I feel like ending my life",

        # BIPOLAR
        "My mood changes suddenly",
        "Sometimes I feel too energetic",
        "I feel excited then suddenly sad",
        "My emotions are unstable",

        # PERSONALITY DISORDER
        "I cannot control my emotions",
        "People say my behavior changes rapidly",
        "I react aggressively sometimes",
        "I struggle with emotional stability"
    ],

    "label": [

        "normal",
        "normal",
        "normal",
        "normal",

        "stress",
        "stress",
        "stress",
        "stress",

        "anxiety",
        "anxiety",
        "anxiety",
        "anxiety",

        "depression",
        "depression",
        "depression",
        "depression",

        "suicidal",
        "suicidal",
        "suicidal",
        "suicidal",

        "bipolar",
        "bipolar",
        "bipolar",
        "bipolar",

        "personality disorder",
        "personality disorder",
        "personality disorder",
        "personality disorder"
    ]
}

df = pd.DataFrame(data)

# =========================================================
# TRAIN MODEL
# =========================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

y = df["label"]

model = LogisticRegression()

model.fit(X, y)

# =========================================================
# GUIDANCE
# =========================================================

guidance = {

    "normal": {
        "message": "You seem emotionally balanced 🌸",
        "activity": "Continue your healthy routine.",
        "tips": [
            "Exercise regularly",
            "Stay productive",
            "Keep learning new things"
        ]
    },

    "stress": {
        "message": "You may be mentally overloaded 💙",
        "activity": "Take a short break and relax.",
        "tips": [
            "Sleep properly",
            "Break work into smaller tasks",
            "Avoid multitasking"
        ]
    },

    "anxiety": {
        "message": "Your text reflects anxiety 🌿",
        "activity": "Take deep breaths slowly.",
        "tips": [
            "Practice meditation",
            "Reduce overthinking",
            "Take screen breaks"
        ]
    },

    "depression": {
        "message": "You may be emotionally exhausted 🌧️",
        "activity": "Talk to someone supportive.",
        "tips": [
            "Avoid isolation",
            "Go outside for sunlight",
            "Maintain sleep schedule"
        ]
    },

    "suicidal": {
        "message": "Please seek support immediately ❤️",
        "activity": "Contact someone you trust right now.",
        "tips": [
            "Do not stay alone",
            "Reach out to family or friends",
            "Seek professional help"
        ]
    },

    "bipolar": {
        "message": "Your text reflects mood instability ⚡",
        "activity": "Maintain a calm and stable routine.",
        "tips": [
            "Track mood changes",
            "Avoid impulsive decisions",
            "Sleep consistently"
        ]
    },

    "personality disorder": {
        "message": "Emotional instability patterns detected 🧠",
        "activity": "Practice mindfulness exercises.",
        "tips": [
            "Write emotional journals",
            "Avoid emotional triggers",
            "Practice calm communication"
        ]
    }
}

# =========================================================
# TITLE
# =========================================================

st.markdown("<div class='title'>🧠 MindCare AI</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Mental Health Emotion Detection System</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About")

st.sidebar.info("""
This project uses:

✅ NLP  
✅ TF-IDF Vectorization  
✅ Logistic Regression  
✅ Mental Health Detection  
✅ Emotional Guidance System
""")

# =========================================================
# INPUT
# =========================================================

user_input = st.text_area(
    "💬 Enter your feelings or thoughts",
    height=180,
    placeholder="Example: I feel mentally exhausted and lonely..."
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        text_vector = vectorizer.transform([user_input])

        prediction = model.predict(text_vector)[0]

        probabilities = model.predict_proba(text_vector)[0]

        confidence = np.max(probabilities) * 100

        # =====================================================
        # RESULTS
        # =====================================================

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class='result-box'>
                    <h2>🧠 Detected Emotion</h2>
                    <h1 style='color:#38bdf8'>{prediction.upper()}</h1>
                    <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            prob_df = pd.DataFrame({
                "Emotion": model.classes_,
                "Probability": probabilities
            })

            fig = px.bar(
                prob_df,
                x="Emotion",
                y="Probability",
                title="Emotion Probabilities"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # GUIDANCE AREA
        # =====================================================

        st.markdown("## 🌈 Emotional Guidance Area")

        data = guidance.get(prediction.lower())

        if data:

            st.markdown(
                f"""
                <div class='tip-box'>
                    <h3>💡 Motivation</h3>
                    <p>{data['message']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class='tip-box'>
                    <h3>🎯 Positive Activity</h3>
                    <p>{data['activity']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 🌿 Wellness Tips")

            for tip in data["tips"]:
                st.success(tip)

        # =====================================================
        # QUOTES
        # =====================================================

        quotes = [
            "Small progress is still progress 🌱",
            "Healing takes time and strength 💙",
            "Your emotions matter 🌸",
            "Rest is productive too 🌙",
            "You survived difficult days before ✨"
        ]

        st.info(random.choice(quotes))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("MindCare AI • NLP Mental Health Detection")
