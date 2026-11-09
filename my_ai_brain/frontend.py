
import streamlit as st
import logging
from datetime import datetime

from my_ai_brain.chat import chat
from my_ai_brain.memory import Memory

try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Personal AI Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        try:
            st.session_state.memory = Memory()
            st.session_state.memory_error = None
        except Exception as e:
            st.session_state.memory = None
            st.session_state.memory_error = str(e)


def process_message(user_input):
    user_input = user_input.strip()

    if not user_input:
        return

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        if st.session_state.memory is None:
            raise RuntimeError(
                "మెమరీ ప్రారంభం కాలేదు. Memory configuration తనిఖీ చేయండి."
            )

        with st.spinner("🧠 AI ఆలోచిస్తోంది..."):
            response = chat(
                st.session_state.memory,
                user_input
            )

        if not isinstance(response, str):
            response = str(response)

        if not response.strip():
            response = "AI నుంచి ఖాళీ సమాధానం వచ్చింది."

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    except Exception as e:
        logger.exception("Chat processing failed")

        error_message = (
            "క్షమించండి! సమాధానం తయారు చేయడంలో సమస్య వచ్చింది.\n\n"
            f"వివరాలు: {str(e)}"
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })


def render_app():

    initialize_state()

    # Custom design
    st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        padding: 12px;
    }
    .subtitle {
        text-align: center;
        color: #888888;
        margin-bottom: 20px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px;
    }
    .status-text {
        font-size: 14px;
        color: #888888;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="main-title">🧠 Personal AI Brain</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">మీ వ్యక్తిగత AI అసిస్టెంట్ | '
        'తెలుగు వాయిస్ & టెక్స్ట్ చాట్</div>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.title("🧠 AI Controls")

        st.markdown("---")

        if st.button("➕ కొత్త చాట్", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ చాట్ హిస్టరీ క్లియర్", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("సిస్టమ్ స్థితి")

        if st.session_state.memory is not None:
            st.success("మెమరీ కనెక్ట్ అయింది")
        else:
            st.error("మెమరీ అందుబాటులో లేదు")
            if st.session_state.memory_error:
                st.caption(st.session_state.memory_error)

        if MIC_AVAILABLE:
            st.success("మైక్ ఫీచర్ అందుబాటులో ఉంది")
        else:
            st.warning("మైక్ ప్యాకేజీ ఇన్‌స్టాల్ కాలేదు")

        st.markdown("---")
        st.caption("Personal AI Brain")
        st.caption("Powered by Local AI + SQLite")

    # Chat history
    st.subheader("💬 మీ సంభాషణ")

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

            if voice_text and voice_text.strip():
                st.info(f"మీరు చెప్పింది: {voice_text}")
                process_message(voice_text)
                st.rerun()

        except Exception as e:
            logger.exception("Microphone error")
            st.warning(f"మైక్ సమస్య: {e}")

    # Text input
    user_input = st.chat_input(
        "మీ ప్రశ్నను ఇక్కడ టైప్ చేయండి..."
    )

    if user_input:
        process_message(user_input)
        st.rerun()


if __name__ == "__main__":
    render_app()
    
