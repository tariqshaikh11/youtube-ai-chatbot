import streamlit as st
from chatbot import YouTubeChatBot


st.set_page_config(page_title="YouTube AI Chatbot", layout="centered")
st.title("🎥 YouTube AI Q&A Assistant")

if "loaded" not in st.session_state:
    st.session_state.loaded = False

if "video_url" not in st.session_state:
    st.session_state.video_url = ""

if "bot" not in st.session_state:
    st.session_state.bot = YouTubeChatBot()

bot = st.session_state.bot

# ---- INPUT YOUTUBE URL ----
st.subheader("🔗 Step 1: Enter YouTube URL")

youtube_url = st.text_input("YouTube URL", placeholder="Paste YouTube link...")
st.text("only english video")
submit_button = st.button("Submit Video")

if submit_button:
    with st.spinner("Processing transcript... ⏳"):
        try:
            msg = bot.load_video(youtube_url)
            st.success(msg)
            st.session_state.loaded = True
        except Exception as e:
            st.error(str(e))


st.markdown("---")

# ---- ASK QUESTION ----
st.subheader("💬 Step 2: Ask a question about the video")

question = st.text_input("Your Question", placeholder="Example: What is the video about?")
ask_button = st.button("Ask")

if ask_button:
    if not st.session_state.loaded:
        st.warning("⚠ Please submit a YouTube video first.")
    else:
        with st.spinner("Thinking... 🤖"):
            answer = bot.ask(question)
            st.success(answer)
