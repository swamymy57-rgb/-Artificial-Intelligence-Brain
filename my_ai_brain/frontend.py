import os
import logging
from pathlib import Path

import streamlit as st

from my_ai_brain.memory import SQLiteMemory
from my_ai_brain.chat import chat
from my_ai_brain.config import Settings
from my_ai_brain.artifacts import load_artifacts
from my_ai_brain.inference import InferenceEngine


# Optional microphone support
try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    speech_to_text = None
    MIC_AVAILABLE = False


# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Streamlit page configuration
st.set_page_config(
    page_title="Personal AI Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load AI model and memory only once
@st.cache_resource(show_spinner=False)
def load_runtime():

    settings = Settings.from_env()

    model_dir = Path(
        getattr(
            settings,
            "model_dir",
            os.getenv("MY_AI_BRAIN_MODEL_DIR", "artifacts")
        )
    )

    database = getattr(
        settings,
        "database",
        os.getenv(
            "MY_AI_BRAIN_DATABASE",
            "/tmp/my-ai-brain-preview.sqlite3"
        )
    )

    device = getattr(settings, "device", "cpu")

    if not model_dir.exists():
        raise FileNotFoundError(
            f"AI model directory not found: {model_dir}"
        )

    model, tokenizer = load_artifacts(
        model_dir,
        device
    )

    engine = InferenceEngine(
        model,
        tokenizer,
        device
    )

    memory = SQLiteMemory(database)

    return engine, memory


# Initialize application state
def initialize_state():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "runtime" not in st.session_state:
        st.session_state.runtime = None

    if "runtime_error" not in st.session_state:
        st.session_state.runtime_error = None

    if st.session_state.runtime is None:

        try:
            with st.spinner("🧠 AI Brain ప్రారంభమవుతోంది..."):

                st.session_state.runtime = load_runtime()

                st.session_state.runtime_error = None

        except Exception as error:

            logger.exception("AI initialization failed")

            st.session_state.runtime_error = str(error)


# Process user message
def process_message(user_input):

    user_input = user_input.strip()

    if not user_input:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    try:

        if st.session_state.runtime is None:
            raise RuntimeError(
                st.session_state.runtime_error
                or "AI Brain ఇంకా ప్రారంభం కాలేదు."
            )

        engine, memory = st.session_state.runtime

        with st.spinner("🧠 AI ఆలోచిస్తోంది..."):

            result = chat(
                engine,
                memory,
                user_input,
                max_new_tokens=40
            )

        if isinstance(result, dict):

            response = result.get(
                "reply",
                "AI నుంచి సమాధానం రాలేదు."
            )

        else:

            response = str(result)

        if not isinstance(response, str):
            response = str(response)

        if not response.strip():
            response = "AI నుంచి ఖాళీ సమాధానం వచ్చింది."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as error:

        logger.exception("Chat processing failed")

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "క్షమించండి! సమాధానం తయారుచేయడంలో సమస్య వచ్చింది.\n\n"
                f"Error: {str(error)}"
            )
        })


# Main application
def render_app():

    initialize_state()

    # Custom CSS
    st.markdown("""
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        padding: 15px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        margin-bottom: 25px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div class="main-title">🧠 Personal AI Brain</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'మీ వ్యక్తిగత AI అసిస్టెంట్ | తెలుగు వాయిస్ & టెక్స్ట్ చాట్'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:

        st.title("🧠 AI Controls")

        st.divider()

        if st.button(
            "➕ కొత్త చాట్",
            use_container_width=True
        ):

            st.session_state.messages = []
            st.rerun()

        if st.button(
            "🗑️ చాట్ హిస్టరీ క్లియర్",
            use_container_width=True
        ):

            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.subheader("సిస్టమ్ స్థితి")

        if st.session_state.runtime is not None:

            st.success("AI Model Loaded")

            st.success("SQLite Memory Connected")

        else:

            st.error("AI ప్రారంభం కాలేదు")

            if st.session_state.runtime_error:

                st.error(
                    st.session_state.runtime_error
                )

        if MIC_AVAILABLE:

            st.success("మైక్రోఫోన్ అందుబాటులో ఉంది")

        else:

            st.warning("మైక్రోఫోన్ ప్యాకేజీ లేదు")

        st.divider()

        st.caption("Personal AI Brain")
        st.caption("Powered by Tiny Transformer + SQLite")

        if st.button(
            "🔄 AI మళ్లీ ప్రారంభించు",
            use_container_width=True
        ):

            st.session_state.runtime = None
            st.session_state.runtime_error = None

            load_runtime.clear()

            st.rerun()

    # Chat section
    st.subheader("💬 మీ సంభాషణ")

    # Display messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Voice input
    if MIC_AVAILABLE:

        st.markdown("### 🎤 వాయిస్ ద్వారా మాట్లాడండి")

        try:

            voice_text = speech_to_text(
                language="te-IN",
                start_prompt="🎤 మాట్లాడటానికి నొక్కండి",
                stop_prompt="⏹️ ఆపండి",
                just_once=True,
                use_container_width=True,
                key="telugu_voice_input"
            )

            if (
                isinstance(voice_text, str)
                and voice_text.strip()
            ):

                st.info(f"మీరు చెప్పింది: {voice_text}")

                process_message(voice_text)

                st.rerun()

        except Exception as error:

            logger.exception("Microphone error")

            st.warning(
                f"మైక్రోఫోన్ సమస్య: {error}"
            )

    # Text chat input
    user_input = st.chat_input(
        "మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి..."
    )

    if user_input:

        process_message(user_input)

        st.rerun()


# Application entry point
if __name__ == "__main__":

    render_app()
