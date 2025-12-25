import os
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from urllib.parse import urlparse, parse_qs


  
def get_video_id(url: str):
        url = url.strip()
        url = url.split("&")[0]

        if not url.startswith("http"):
            url = "https://" + url

        parsed = urlparse(url)

        host = parsed.hostname or ""
        path = parsed.path

        # watch url
        if "youtube.com" in host and path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]

        # shorts
        if "youtube.com" in host and path.startswith("/shorts/"):
            return path.split("/shorts/")[1]

        # embed
        if "youtube.com" in host and path.startswith("/embed/"):
            return path.split("/embed/")[1]

        # youtu.be short
        if "youtu.be" in host:
            return path.lstrip("/")

        return None
class YouTubeChatBot:

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        self.vector_store = None
        self.retriever = None
        self.main_chain = None
        self.loaded = False

    

    

    def load_video(self, video_url):

        try:
            
            video_id = get_video_id(video_url)
            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
        except:
            raise Exception("❌ Invalid YouTube URL")

        #transcript = YouTubeTranscriptApi.fetch(video_id)

        # Convert transcript to plain text
        formatter = JSONFormatter()
        json_text = formatter.format_transcript(transcript)
        arr = json.loads(json_text)
        full_text = " ".join(item["text"] for item in arr)

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([full_text])

        # Embeddings
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_documents(chunks, embedding)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # Setup RAG chain
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY using this transcript:

{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        parallel_chain = RunnableParallel({
                "context": self.retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
        })

        self.main_chain = parallel_chain | prompt | self.llm | StrOutputParser()
        self.loaded = True
        return "✔ Transcript successfully indexed! "

    def ask(self, question):
        if not self.loaded:
            return "⚠ You must submit a video first."
    
        return self.main_chain.invoke(question)

