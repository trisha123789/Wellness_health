import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px
import pandas as pd
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

.main {
    background-color: #0f172a;
    color: white;
}

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

.title {
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle {
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
}

.prediction-box {
    padding:20px;
    border-radius:20px;
    background:linear-gradient(135deg,#1e293b,#0f172a);
    box-shadow:0px 0px 20px rgba(0,0,0,0.5);
}

.tip-box {
    padding:18px;
    border-radius:18px;
    background:#111827;
    border-left:5px solid #38bdf8;
    margin-top:15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL + TOKENIZER
# =========================================================

model = load_model("mental_health_rnn.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# =========================================================
# LABEL GUIDANCE
# =========================================================

guidance = {

    "normal": {
        "message": "You seem emotionally balanced today 🌸",
        "activity": "Maintain your routine and spend time doing something creative.",
        "tips": [
            "Stay hydrated",
            "Exercise for 20 minutes",
            "Keep a gratitude journal"
        ]
    },

    "anxiety": {
        "message": "Your text reflects signs of anxiety. Slow down and breathe 🌿",
        "activity": "Take a short walk and avoid overstimulating environments.",
        "tips": [
            "Practice deep breathing",
            "Reduce screen time",
            "Talk with someone you trust"
        ]
    },

    "stress": {
        "message": "You may be mentally overwhelmed right now 💙",
        "activity": "Take a short break and listen to calming music.",
        "tips": [
            "Prioritize sleep",
            "Break tasks into smaller goals",
            "Avoid multitasking"
        ]
    },

    "depression": {
        "message": "Your text may indicate emotional exhaustion or sadness 🌧️",
        "activity": "Spend time outdoors or with supportive people.",
        "tips": [
            "Do not isolate yourself",
            "Maintain a regular sleep schedule",
            "Seek professional guidance if feelings persist"
        ]
    },

    "suicidal": {
        "message": "Your emotional state may need urgent support ❤️",
        "activity": "Please reach out to someone immediately.",
        "tips": [
            "Contact a trusted person",
            "Avoid staying alone",
            "Seek professional mental health support"
        ]
    },

    "personality disorder": {
        "message": "Your text shows emotional instability patterns 🧠",
        "activity": "Maintain a calm environment and healthy communication.",
        "tips": [
            "Practice mindfulness",
            "Maintain emotional journals",
            "Consider therapy support"
        ]
    },

    "bipolar": {
        "message": "Mood fluctuations may be reflected in your text ⚡",
        "activity": "Try maintaining a stable daily routine.",
        "tips": [
            "Track mood patterns",
            "Sleep consistently",
            "Avoid impulsive decisions"
        ]
    }
}

# =========================================================
# TITLE
# =========================================================

st.markdown("<div class='title'>🧠 MindCare AI</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>RNN + NLP Mental Health Emotion Detection System</div>",
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📌 About")

st.sidebar.write("""
This project uses:

✅ NLP  
✅ Tokenization  
✅ RNN Deep Learning  
✅ Emotional Classification  
✅ Real-Time Guidance System  
""")

st.sidebar.success("Built with Streamlit + TensorFlow")

# =========================================================
# INPUT AREA
# =========================================================

user_input = st.text_area(
    "💬 Enter your thoughts or feelings",
    height=180,
    placeholder="Example: I feel lonely and mentally exhausted..."
)

# =========================================================
# PREDICTION
# =========================================================

if st.button("🔍 Analyze Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Tokenization
        sequence = tokenizer.texts_to_sequences([user_input])

        # Padding
        padded = pad_sequences(sequence, maxlen=100)

        # Prediction
        prediction = model.predict(padded)

        predicted_class = np.argmax(prediction)

        emotion = label_encoder.inverse_transform([predicted_class])[0]

        confidence = float(np.max(prediction)) * 100

        # =====================================================
        # DISPLAY RESULT
        # =====================================================

        st.markdown("---")

        col1, col2 = st.columns([1,1])

        with col1:

            st.markdown(
                f"""
                <div class='prediction-box'>
                <h2>🧠 Detected Emotion</h2>
                <h1 style='color:#38bdf8'>{emotion.upper()}</h1>
                <h3>Confidence: {confidence:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            chart_df = pd.DataFrame({
                "Emotion": label_encoder.classes_,
                "Probability": prediction[0]
            })

            fig = px.bar(
                chart_df,
                x="Emotion",
                y="Probability",
                title="Prediction Probabilities"
            )

            st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # GUIDANCE AREA
        # =====================================================

        st.markdown("## 🌈 Emotional Guidance Area")

        data = guidance.get(emotion.lower())

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

            for tip in data['tips']:
                st.success(tip)

        # =====================================================
        # RANDOM MOTIVATIONAL QUOTE
        # =====================================================

        quotes = [
            "Small progress is still progress.",
            "You survived difficult days before.",
            "Your current situation is not your final destination.",
            "Rest is productive too.",
            "Healing takes time and strength."
        ]

        st.info(f"✨ {random.choice(quotes)}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("MindCare AI • RNN + NLP Mental Health Detection")