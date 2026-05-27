
import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import random

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
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 15px !important;
    border: 1px solid #475569 !important;
    font-size: 18px !important;
}

.result-box {
    background: #1e293b;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}

.tip-box {
    background: #111827;
    padding: 18px;
    border-radius: 16px;
    margin-top: 15px;
    border-left: 5px solid #38bdf8;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = pickle.load(open("mental_health_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# =========================================================
# GUIDANCE DATA
# =========================================================

guidance = {

    "normal": {
        "message": "You seem emotionally balanced today 🌸",
        "activity": "Spend time learning something creative.",
        "tips": [
            "Exercise regularly",
            "Maintain healthy sleep",
            "Stay socially connected"
        ]
    },

    "anxiety": {
        "message": "Your text reflects anxiety signs 🌿",
        "activity": "Take deep breaths and relax for a few minutes.",
        "tips": [
            "Avoid overthinking",
            "Practice meditation",
            "Reduce screen exposure"
        ]
    },

    "stress": {
        "message": "You may be mentally overloaded 💙",
        "activity": "Take a short walk or listen to calm music.",
        "tips": [
            "Break tasks into small steps",
            "Sleep properly",
            "Avoid multitasking"
        ]
    },

    "depression": {
        "message": "Your text may indicate sadness 🌧️",
        "activity": "Talk to someone you trust.",
        "tips": [
            "Do not isolate yourself",
            "Get sunlight exposure",
            "Seek professional support if needed"
        ]
    },

    "suicidal": {
        "message": "Please seek immediate emotional support ❤️",
        "activity": "Contact someone close to you right now.",
        "tips": [
            "Avoid staying alone",
            "Call a trusted person",
            "Reach out to mental health professionals"
        ]
    },

    "bipolar": {
        "message": "Your text reflects mood fluctuations ⚡",
        "activity": "Maintain a stable daily routine.",
        "tips": [
            "Track your mood",
            "Maintain sleep schedule",
            "Avoid impulsive decisions"
        ]
    },

    "personality disorder": {
        "message": "Your text shows emotional instability patterns 🧠",
        "activity": "Practice mindfulness and emotional regulation.",
        "tips": [
            "Write journals",
            "Avoid emotional triggers",
            "Maintain calm communication"
        ]
    }
}

# =========================================================
# TITLE
# =========================================================

st.markdown("<div class='title'>🧠 MindCare AI</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>Mental Health Emotion Detection using NLP</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📌 About Project")

st.sidebar.info("""
This AI system detects:

✅ Stress  
✅ Anxiety  
✅ Depression  
✅ Bipolar  
✅ Suicidal Thoughts  
✅ Personality Disorder  
✅ Normal Emotion
""")

# =========================================================
# INPUT
# =========================================================

user_input = st.text_area(
    "💬 Enter your feelings or thoughts",
    height=180,
    placeholder="Example: I feel emotionally exhausted and lonely..."
)

# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Transform text
        text_vector = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Probability
        probabilities = model.predict_proba(text_vector)[0]

        confidence = np.max(probabilities) * 100

        # =====================================================
        # RESULT DISPLAY
        # =====================================================

        st.markdown("---")

        col1, col2 = st.columns([1,1])

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
        # GUIDANCE SECTION
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
        # RANDOM QUOTES
        # =====================================================

        quotes = [
            "Small progress is still progress 🌱",
            "Healing takes time and courage 💙",
            "Your emotions are valid 🌸",
            "Rest is productive too 🌙",
            "You survived difficult days before ✨"
        ]

        st.info(random.choice(quotes))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.caption("MindCare AI • NLP Mental Health Detection")
