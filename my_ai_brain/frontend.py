import os
import json
import html
import logging
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from my_ai_brain.memory import SQLiteMemory
from my_ai_brain.chat import chat
from my_ai_brain.config import Settings
from my_ai_brain.artifacts import load_artifacts
from my_ai_brain.inference import InferenceEngine


# Optional microphone
try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    speech_to_text = None
    MIC_AVAILABLE = False


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="MY AI BRAIN",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# -----------------------------
# AI MODEL LOADER
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_runtime():

    settings = Settings.from_env()

    model_dir = Path(settings.model_dir).expanduser().resolve()
    database = settings.database
    device = settings.device

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"AI model directory not found: {model_dir}"
        )

    model, tokenizer = load_artifacts(model_dir, device)

    engine = InferenceEngine(model, tokenizer, device)
    memory = SQLiteMemory(database)

    return engine, memory


# -----------------------------
# SESSION STATE
# -----------------------------

def initialize_state():

    defaults = {
        "messages": [],
        "runtime": None,
        "runtime_error": None,
        "uploaded_images": [],
        "input_text": ""
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.runtime is None:

        try:
            with st.spinner("🧠 AI Brain ప్రారంభమవుతోంది..."):
                st.session_state.runtime = load_runtime()
                st.session_state.runtime_error = None

        except Exception as error:
            logger.exception("AI initialization failed")
            st.session_state.runtime_error = str(error)


# -----------------------------
# SPEAKER / TEXT TO SPEECH
# -----------------------------

def render_speaker(text):

    safe_text = json.dumps(str(text), ensure_ascii=False)

    components.html(
        f"""
        <div style="display:flex;align-items:center;">
          <button id="speakBtn"
            style="
              background:#292929;
              color:white;
              border:none;
              border-radius:20px;
              padding:7px 13px;
              cursor:pointer;
              font-size:14px;">
            🔊 వినండి
          </button>
        </div>

        <script>
        const message = {safe_text};
        const button = document.getElementById("speakBtn");

        button.onclick = function() {{
            if (!window.speechSynthesis) {{
                button.innerText = "వాయిస్ అందుబాటులో లేదు";
                return;
            }}

            window.speechSynthesis.cancel();

            const speech = new SpeechSynthesisUtterance(message);
            speech.lang = "te-IN";
            speech.rate = 0.9;

            window.speechSynthesis.speak(speech);
            button.innerText = "🔊 వింటున్నారు...";
        }};
        </script>
        """,
        height=48
    )


# -----------------------------
# PROCESS MESSAGE
# -----------------------------

def process_message(user_input):

    user_input = user_input.strip()

    if not user_input:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    if st.session_state.runtime is None:

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "AI Model ప్రారంభం కాలేదు.\n\n"
                + str(st.session_state.runtime_error)
            )
        })
        return

    try:
        engine, memory = st.session_state.runtime

        with st.spinner("🧠 AI ఆలోచిస్తోంది..."):

            result = chat(
                engine,
                memory,
                user_input,
                max_new_tokens=100
            )

        if isinstance(result, dict):
            response = result.get(
                "reply",
                result.get("text", "సమాధానం రాలేదు.")
            )
        else:
            response = str(result)

        response = str(response).strip()

        if not response:
            response = "AI నుంచి ఖాళీ సమాధానం వచ్చింది."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as error:

        logger.exception("Chat error")

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"సమస్య వచ్చింది: {error}"
        })


# -----------------------------
# DESIGN
# -----------------------------

def add_design():

    st.markdown("""
    <style>

    .block-container {
        max-width: 850px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .brain-header {
        text-align:center;
        padding: 12px;
    }

    .brain-logo {
        font-size: 42px;
        background: linear-gradient(135deg,#8b5cf6,#06b6d4);
        width: 68px;
        height: 68px;
        border-radius: 22px;
        margin: auto;
        display:flex;
        align-items:center;
        justify-content:center;
    }

    .brain-title {
        font-size: 25px;
        font-weight: 800;
        margin-top: 10px;
    }

    .brain-subtitle {
        color: #999;
        font-size: 13px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 15px;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# MAIN APP
# -----------------------------

def render_app():

    initialize_state()
    add_design()

    # Header
    st.markdown("""
    <div class="brain-header">
        <div class="brain-logo">🧠</div>
        <div class="brain-title">MY AI BRAIN</div>
        <div class="brain-subtitle">
            Your Personal Intelligent Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Sidebar
    with st.sidebar:

        st.title("🧠 AI Settings")

        if st.session_state.runtime is not None:
            st.success("AI Model Loaded")
            st.success("Memory Connected")
        else:
            st.error("AI Model Not Loaded")
            st.error(st.session_state.runtime_error or "")

        st.divider()

        if st.button("➕ కొత్త చాట్", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ చాట్ క్లియర్", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 AI Restart", use_container_width=True):
            st.session_state.runtime = None
            st.session_state.runtime_error = None
            load_runtime.clear()
            st.rerun()

        st.divider()
        st.caption("MY AI BRAIN")
        st.caption("Tiny Transformer + SQLite Memory")

    # Chat history
    st.subheader("💬 సంభాషణ")

    if not st.session_state.messages:
        st.info(
            "నమస్కారం! 👋 మీ AI Brain కు స్వాగతం. "
            "మీ ప్రశ్నను టైప్ చేయండి లేదా మాట్లాడండి."
        )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message["role"] == "assistant":
                render_speaker(message["content"])

    st.divider()

    # Image upload
    st.markdown("### 📎 ఫోటోలు జోడించండి")

    uploaded_files = st.file_uploader(
        "➕ ఫోటోలను ఎంచుకోండి",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="image_upload"
    )

    if uploaded_files:

        st.session_state.uploaded_images = uploaded_files

        for uploaded in uploaded_files:
            st.image(
                uploaded,
                caption=uploaded.name,
                use_container_width=True
            )

        st.caption(
            "ఫోటోలు జోడించబడ్డాయి. "
            "ప్రస్తుత టెక్స్ట్ మోడల్ వాటిని విశ్లేషించలేదు."
        )

    # Voice input
    st.markdown("### 🎤 వాయిస్ ఇన్‌పుట్")

    voice_text = None

    if MIC_AVAILABLE:

        try:
            voice_text = speech_to_text(
                language="te-IN",
                start_prompt="🎤 మాట్లాడండి",
                stop_prompt="⏹️ ఆపండి",
                just_once=True,
                use_container_width=True,
                key="telugu_voice"
            )

        except Exception as error:
            st.warning(f"మైక్రోఫోన్ సమస్య: {error}")

    else:
        st.warning(
            "మైక్రోఫోన్ కోసం streamlit-mic-recorder "
            "ప్యాకేజీని requirements.txt లో చేర్చాలి."
        )

    if isinstance(voice_text, str) and voice_text.strip():

        st.info(f"మీరు చెప్పింది: {voice_text}")

        process_message(voice_text)
        st.rerun()

    # Text input
    st.markdown("### ✍️ మీ ప్రశ్న")

    with st.form("message_form", clear_on_submit=True):

        user_input = st.text_area(
            "మీ సందేశం",
            placeholder="ఇక్కడ మీ ప్రశ్న టైప్ చేయండి...",
            height=85,
            label_visibility="collapsed"
        )

        send = st.form_submit_button(
            "➤ పంపండి",
            use_container_width=True
        )

        if send and user_input.strip():
            process_message(user_input)
            st.rerun()


if __name__ == "__main__":
    render_app()
    
