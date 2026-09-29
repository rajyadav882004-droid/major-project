
import streamlit as st
from pathlib import Path

# ============================================================
# DEPRESSION DETECTION USING MACHINE LEARNING
# Classic, eye-catching Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Depression Detection Using Machine Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Theme / classic UI
# -----------------------------
st.markdown("""
<style>
/* General */
.stApp {
    background: #f5f1e8;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Typography */
h1, h2, h3 {
    color: #172033 !important;
}
p, label, .stMarkdown {
    color: #303746;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #172033;
}
section[data-testid="stSidebar"] * {
    color: #f5f1e8 !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #b18a3a;
    min-height: 44px;
    font-weight: 700;
}
.stButton > button:hover {
    border-color: #8d6b2c;
}

/* Radio cards */
div[role="radiogroup"] {
    gap: 8px;
}
div[role="radiogroup"] label {
    background: #fffdf8;
    border: 1px solid #d8cfbd;
    border-radius: 8px;
    padding: 10px 14px;
}
div[role="radiogroup"] label:hover {
    border-color: #b18a3a;
}

/* Progress */
.stProgress > div > div > div > div {
    background: #b18a3a;
}

/* Metric */
div[data-testid="stMetric"] {
    background: #fffdf8;
    border: 1px solid #d8cfbd;
    padding: 16px;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #d8cfbd;
}

/* Quote / info */
blockquote {
    border-left: 4px solid #b18a3a;
    background: #fffdf8;
    padding: 12px 18px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state
# -----------------------------
if "answers" not in st.session_state:
    st.session_state.answers = [None] * 20

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "result" not in st.session_state:
    st.session_state.result = None

# -----------------------------
# Assessment data
# -----------------------------
questions = [
    "During the past week, how often have you felt unusually sad or low?",
    "During the past week, how often have you felt less interested in activities you normally enjoy?",
    "During the past week, how often have you felt tired or lacking energy?",
    "During the past week, how often have you felt hopeful about the future?",
    "During the past week, how often have you had difficulty concentrating?",
    "During the past week, how often have you felt lonely?",
    "During the past week, how often have you had difficulty sleeping or sleeping too much?",
    "During the past week, how often have you felt happy or positive?",
    "During the past week, how often have you felt that everyday tasks were difficult to manage?",
    "During the past week, how often have you felt worried or anxious?",
    "During the past week, how often have you experienced changes in your appetite?",
    "During the past week, how often have you enjoyed spending time with other people?",
    "During the past week, how often have you felt that you were not performing as well as usual?",
    "During the past week, how often have you felt emotionally overwhelmed?",
    "During the past week, how often have you had difficulty getting started with normal activities?",
    "During the past week, how often have you felt confident in your ability to handle daily problems?",
    "During the past week, how often have you felt discouraged?",
    "During the past week, how often have you felt connected to friends, family, or other people?",
    "During the past week, how often have you found it difficult to enjoy things that normally make you happy?",
    "During the past week, how often have you felt that you needed additional emotional support?"
]

options = ["Never", "Sometimes", "Often", "Almost always"]

# Questions where a positive answer indicates lower symptom burden.
# This is custom educational scoring, NOT official CES-D scoring.
reverse_questions = {3, 7, 11, 15, 17}

def calculate_score():
    total = 0
    for i, answer in enumerate(st.session_state.answers):
        if answer is None:
            continue
        value = answer
        if i in reverse_questions:
            value = 3 - value
        total += value
    return total

def reset_assessment():
    st.session_state.answers = [None] * 20
    st.session_state.current_question = 0
    st.session_state.result = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🧠 MindCheck")
    st.caption("Machine Learning • NLP • Self-Assessment")
    st.divider()

    st.markdown("### Project Modules")
    st.markdown("📋 **20-Question Assessment**")
    st.markdown("🤖 **DistilBERT Text Analysis**")
    st.markdown("📊 **Result & Score Analysis**")
    st.markdown("🏥 **Support Resources**")

    st.divider()
    st.markdown("### About")
    st.caption(
        "An educational college project demonstrating questionnaire-based "
        "assessment and machine-learning text classification."
    )

    if st.button("↻ Start New Assessment", use_container_width=True):
        reset_assessment()
        st.rerun()

# -----------------------------
# Header
# -----------------------------
st.markdown("# 🧠 Depression Detection Using Machine Learning")
st.markdown(
    "### A classic machine-learning based mental wellness assessment system"
)

st.info(
    "This application is for educational and screening purposes only. "
    "It does not provide a medical diagnosis."
)

st.divider()

# -----------------------------
# Result view
# -----------------------------
if st.session_state.result is not None:
    score = st.session_state.result

    st.markdown("## 📊 Assessment Result")

    if score <= 15:
        title = "Lower Reported Symptom Burden"
        message = (
            "Your responses indicate a relatively lower level of reported "
            "symptoms in this educational assessment."
        )
        icon = "🌿"
    elif score <= 30:
        title = "Moderate Reported Symptom Burden"
        message = (
            "Your responses indicate a moderate level of reported symptoms. "
            "Consider speaking with someone you trust if these feelings are "
            "affecting your daily life."
        )
        icon = "💬"
    else:
        title = "Higher Reported Symptom Burden"
        message = (
            "Your responses indicate a higher level of reported symptoms. "
            "Consider talking with a qualified mental-health professional "
            "for proper assessment and support."
        )
        icon = "❤️"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Assessment Score", f"{score} / 60")
    with col2:
        st.metric("Questions Completed", "20 / 20")
    with col3:
        st.metric("Assessment Type", "Educational")

    st.markdown("---")
    st.markdown(f"## {icon} {title}")
    st.write(message)

    if score > 30:
        st.warning(
            "If you feel that you may be in immediate danger or might hurt "
            "yourself, do not rely on this application. Contact emergency "
            "services or seek immediate help from a qualified professional."
        )

        st.markdown("### 🏥 Mental-Health Support — India")
        st.write(
            "You can contact **Tele-MANAS**, the Government of India's "
            "24/7 mental-health support service:"
        )
        st.success("📞 Tele-MANAS: **14416** or **1800-89-14416**")

        st.markdown("#### Find a nearby mental-health service")
        location = st.text_input(
            "Enter your city, area or PIN code",
            placeholder="Example: Ghaziabad, Uttar Pradesh"
        )
        if location.strip():
            query = location.strip().replace(" ", "+")
            st.markdown(
                f"[🔎 Search nearby mental-health hospitals / psychiatrists]"
                f"(https://www.google.com/maps/search/mental+health+hospital+"
                f"near+{query})"
            )

    st.markdown("---")
    st.markdown("### 🤖 Optional: DistilBERT Text Analysis")
    st.write(
        "The trained DistilBERT model can separately classify a piece of "
        "text using the social-media dataset used for this project. "
        "Its prediction should not be treated as a medical diagnosis."
    )

    text = st.text_area(
        "Write a short sentence or paragraph:",
        placeholder="Example: I have been feeling very sad and lonely lately."
    )

    if st.button("Analyze Text", type="primary"):
        model_dir = Path("models/distilbert_depression")

        if not model_dir.exists():
            st.error("Trained DistilBERT model was not found.")
        elif not text.strip():
            st.warning("Please enter some text first.")
        else:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification

                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_dir)
                )
                model.eval()

                encoded = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                )

                with torch.no_grad():
                    output = model(**encoded)
                    probabilities = torch.softmax(output.logits, dim=1)[0]
                    prediction = int(torch.argmax(probabilities))
                    confidence = float(probabilities[prediction]) * 100

                labels = {
                    0: "Not Depressed",
                    1: "Moderately Depressed",
                    2: "Severely Depressed"
                }

                st.success(
                    f"Model prediction: **{labels[prediction]}**  \n"
                    f"Confidence: **{confidence:.2f}%**"
                )

            except Exception as e:
                st.error(f"Model analysis failed: {e}")

    st.divider()
    if st.button("← Take Assessment Again", type="primary"):
        reset_assessment()
        st.rerun()

else:
    # -----------------------------
    # Assessment view
    # -----------------------------
    q = st.session_state.current_question
    answered = sum(a is not None for a in st.session_state.answers)

    st.markdown("## 📋 20-Question Assessment")
    st.write(
        "Answer each question based on how you have generally felt during "
        "the **past week**."
    )

    st.progress(answered / 20)
    st.caption(f"Progress: {answered} of 20 questions answered")

    st.markdown("---")

    st.markdown(f"### Question {q + 1} of 20")
    st.markdown(f"**{questions[q]}**")

    current_answer = st.session_state.answers[q]

    # Placeholder avoids the None/list-index problem on older Streamlit versions.
    radio_options = ["-- Select an answer --"] + options
    default_index = 0 if current_answer is None else current_answer + 1

    selected = st.radio(
        "Choose one answer:",
        radio_options,
        index=default_index,
        key=f"question_{q}"
    )

    if selected != "-- Select an answer --":
        st.session_state.answers[q] = options.index(selected)

    st.markdown("---")

    left, mid, right = st.columns([1, 1, 1])

    with left:
        if q > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.current_question -= 1
                st.rerun()

    with mid:
        st.write("")

    with right:
        if q < 19:
            if st.button("Next →", type="primary", use_container_width=True):
                if st.session_state.answers[q] is None:
                    st.warning("Please select an answer before continuing.")
                else:
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            if st.button("✓ Calculate Result", type="primary", use_container_width=True):
                if any(a is None for a in st.session_state.answers):
                    st.warning("Please answer all 20 questions before calculating your result.")
                else:
                    st.session_state.result = calculate_score()
                    st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How the score works")
    st.write(
        "Each response contributes between 0 and 3 points. Some positively "
        "worded questions are reverse-scored so that the total score moves "
        "consistently with reported symptom burden. The maximum score is 60."
    )

    st.caption(
        "Important: this is a custom educational self-assessment created for "
        "the project. It is not an officially validated diagnostic instrument."
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Depression Detection Using Machine Learning • College Project • "
    "Educational Use Only"
)
