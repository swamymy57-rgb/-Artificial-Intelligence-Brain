import streamlit as st
from my_ai_brain.chat import chat
from my_ai_brain.memory import get_db_connection

def render_app():
    # పేజీ హెడ్డర్ మరియు టైటిల్
    st.set_page_config(
        page_title="Personal AI Brain",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("🧠 Personal AI Brain")
    st.markdown("లోకల్ మోడల్ మరియు SQLite మెమరీ ఆధారిత AI అసిస్టెంట్.")

    # సైడ్‌బార్‌లో కొత్త చాట్ మరియు సెట్టింగ్స్
    with st.sidebar:
        st.header("నియంత్రణలు (Controls)")
        if st.button("➕ కొత్త చాట్ (New Chat)", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.markdown("---")
        st.subheader("స్థితి (Status)")
        st.info("బ్యాక్‌ఎండ్ లోకల్ మోడల్ మరియు మెమరీ యాక్టివ్‌గా ఉన్నాయి.")

    # చాట్ హిస్టరీని సెషన్ స్టేట్‌లో నిర్వహించడం
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # పాత మెసేజ్‌లను స్క్రీన్‌పై చూపించడం
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # యూజర్ ఇన్‌పుట్ బాక్స్ (చాట్ ఇన్‌పుట్)
    user_input = st.chat_input("మీ సందేశాన్ని ఇక్కడ టైప్ చేయండి...")

    if user_input:
        # యూజర్ మెసేజ్‌ను అపెండ్ చేసి చూపించడం
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI రెస్పాన్స్ జనరేట్ చేయడం
        with st.chat_message("assistant"):
            with st.spinner("ఆలోచిస్తోంది..."):
                try:
                    # బ్యాక్‌ఎండ్ చాట్ ఫంక్షన్‌ని పిలవడం
                    response = chat(user_input)
                    st.markdown(response)
                    # అసిస్టెంట్ రెస్పాన్స్‌ను హిస్టరీలో భద్రపరచడం
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"ఎర్రర్ వచ్చింది: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                  
