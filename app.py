import time
import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    font-size: 55px;
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
    box-shadow: 0px 0px 15px rgba(0,0,0,0.5);
}

.tip-box {
    background: #111827;
    padding: 18px;
    border-radius: 16px;
    margin-top: 15px;
    border-left: 5px solid #38bdf8;
}

.stButton>button {
    background: linear-gradient(90deg,#38bdf8,#0ea5e9);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

[data-testid="metric-container"] {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 15px;
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
# MODEL TRAINING
# =========================================================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["text"])

y = df["label"]

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

# =========================================================
# GUIDANCE DATA
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

st.markdown(
    "<div class='title'>🧠 MindCare AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Mental Health Emotion Detection System</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🧠 MindCare Dashboard")

st.sidebar.success("AI Mental Health Assistant")

st.sidebar.markdown("---")

st.sidebar.markdown("## 📌 Features")

st.sidebar.write("""
✅ NLP Text Analysis  
✅ TF-IDF Vectorization  
✅ Mental Health Detection  
✅ Emotional Guidance  
✅ Probability Analytics  
✅ Wellness Tips  
""")

st.sidebar.markdown("---")

st.sidebar.markdown("## 📊 Dataset Size")

st.sidebar.metric("Training Samples", len(df))
st.sidebar.metric("Emotion Classes", len(df['label'].unique()))

st.sidebar.markdown("---")

st.sidebar.markdown("## 💻 Technologies")

st.sidebar.code("""
Python
Streamlit
Scikit-Learn
Plotly
Pandas
""")

# =========================================================
# USER INPUT
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

        # Loading animation
        with st.spinner("Analyzing emotional patterns..."):
            time.sleep(1)

        # Vectorize input
        text_vector = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Probabilities
        probabilities = model.predict_proba(text_vector)[0]

        confidence = np.max(probabilities) * 100

        # =====================================================
        # METRICS
        # =====================================================

        st.markdown("---")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric("🧠 Emotion", prediction.upper())

        with metric2:
            st.metric("🎯 Confidence", f"{confidence:.2f}%")

        with metric3:
            risk_level = (
                "High"
                if prediction in ["suicidal", "depression"]
                else "Moderate"
            )
            st.metric("⚠️ Emotional Intensity", risk_level)

        # =====================================================
        # CONFIDENCE BAR
        # =====================================================

        st.markdown("### 🔥 Confidence Meter")

        st.progress(float(confidence / 100))

        # =====================================================
        # RESULT SECTION
        # =====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                f"""
                <div class='result-box'>
                    <h2>🧠 Detected Emotion</h2>
                    <h1 style='color:#38bdf8'>
                        {prediction.upper()}
                    </h1>
                    <h3>Confidence Score: {confidence:.2f}%</h3>
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
                text_auto=True,
                title="📊 Emotion Probability Distribution"
            )

            fig.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font_color="white"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # PIE CHART
        # =====================================================

        st.markdown("## 🥧 Probability Distribution Pie Chart")

        pie_fig = px.pie(
            prob_df,
            names="Emotion",
            values="Probability",
            hole=0.45
        )

        pie_fig.update_layout(
            paper_bgcolor="#111827",
            font_color="white"
        )

        st.plotly_chart(pie_fig, use_container_width=True)

        # =====================================================
        # EMOTIONAL GUIDANCE
        # =====================================================

        st.markdown("## 🌈 Emotional Guidance Area")

        data = guidance.get(prediction.lower())

        if data:

            guide1, guide2 = st.columns(2)

            with guide1:

                st.markdown(
                    f"""
                    <div class='tip-box'>
                        <h3>💡 Motivation</h3>
                        <p>{data['message']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with guide2:

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
        # WELLNESS SCORE
        # =====================================================

        st.markdown("## ❤️ Emotional Wellness Score")

        if prediction == "normal":
            wellness_score = 95
        else:
            wellness_score = max(10, int(100 - confidence))

        st.slider(
            "Current Emotional Wellness",
            0,
            100,
            wellness_score,
            disabled=True
        )

        # =====================================================
        # MOTIVATIONAL QUOTES
        # =====================================================

        quotes = [
            "Small progress is still progress 🌱",
            "Healing takes time and strength 💙",
            "Your emotions matter 🌸",
            "Rest is productive too 🌙",
            "You survived difficult days before ✨",
            "Growth happens slowly but continuously 🚀",
            "Every difficult phase teaches resilience 🌿"
        ]

        st.info(random.choice(quotes))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("MindCare AI • NLP Mental Health Detection")
