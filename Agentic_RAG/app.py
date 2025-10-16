import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.agents import initialize_agent, Tool, AgentType
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
import requests

# --- Load Environment Variables ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="🎓 NareshIT AI Assistant", layout="wide")

# --- Logo + Title ---
logo = Image.open("images/images-logo.png")
st.image(logo, width=180)

st.markdown("""
    <div style="text-align: center;">
        <h1>NareshIT AI Assistant 🤖🎓</h1>
        <h4 style="color: gray;">Ask about courses, trainers, syllabus, and reviews</h4>
    </div>
""", unsafe_allow_html=True)

# --- Language Selection ---
language_codes = {
    "English": "en", "Hindi": "hi", "Telugu": "te", "Tamil": "ta",
    "Kannada": "kn", "Marathi": "mr", "Gujarati": "gu", "Bengali": "bn",
    "Punjabi": "pa", "Malayalam": "ml"
}
st.markdown("### 🌐 Choose your language")
selected_language = st.selectbox("Language", list(language_codes.keys()))
lang_code = language_codes[selected_language]

# --- Load PDF & Build Vector DB ---
pdf_path = "nareshiT_context2.pdf"

with st.spinner("🔍 Loading NareshIT courses & trainers..."):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings)
    vectordb.persist()

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={"k": 10}),
        llm=ChatGroq(api_key=GROQ_API_KEY,
                     model="llama-3.3-70b-versatile",
                     temperature=1.0)
    )

# --- Friendly Prompt ---
nareshit_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are NareshIT’s friendly AI assistant 🤖🎓.  
Use the retrieved context as the primary source. Provide short, friendly, and clear answers with emojis 🎯🚀⭐.  

Context: {context}
Question: {question}
Answer:
"""
)

# --- LLM Setup ---
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile", temperature=1.0)

# --- RAG QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": nareshit_prompt},
)

# --- Web Scraper Tool ---
course_schedule_url = "https://nareshit.in/course-schedule/"

def fetch_trainer_from_web(query: str):
    try:
        resp = requests.get(course_schedule_url, timeout=10)
        if resp.status_code != 200:
            return f"Could not reach schedule page (status code {resp.status_code})."

        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.find_all("tr")

        matches = []
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if any(query.lower() in c.lower() for c in cols):
                course = cols[0] if len(cols) > 0 else "Unknown"
                trainer = cols[1] if len(cols) > 1 else "Unknown"
                date = cols[2] if len(cols) > 2 else "Unknown"
                time = cols[3] if len(cols) > 3 else "Unknown"
                matches.append(
                    f"Course: {course}\nTrainer: {trainer}\nStart Date: {date}\nTime: {time}\n"
                )

        if not matches:
            return f"No matching course or trainer found for '{query}' on NareshIT schedule."
        return "\n---\n".join(matches)
    except Exception as e:
        return f"Error fetching schedule: {str(e)}"

trainer_tool = Tool(
    name="Course Schedule Web Fetcher",
    func=fetch_trainer_from_web,
    description="Fetch latest batch schedules, courses, or trainer info from NareshIT website."
)

# --- Wrap RAG QA as Tool ---
retriever_tool = Tool(
    name="Document Retriever",
    func=lambda q: qa_chain.invoke({"query": q})["result"] if isinstance(qa_chain.invoke({"query": q}), dict) else qa_chain.invoke({"query": q}),
    description="Answer questions using NareshIT training documents."
)

# --- Initialize Agent ---
tools = [retriever_tool, trainer_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

system_instruction = """
You are a helpful College Administrator at Naresh IT.
- If the student asks about sessions, trainers, or course schedules, ALWAYS call the Course Schedule Web Fetcher tool.
- If the student asks about institute profile, history, placement, or syllabus, use the Document Retriever.
- Always give a friendly, student-facing response with final information.
"""

st.success("✅ NareshIT Knowledge Base & Agent Ready!")

# --- Session State for History ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# --- Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 📜 Previous Questions & Answers")
    if st.button("🗑️ Clear History"):
        st.session_state.qa_history = []
        st.success("✅ Chat history cleared!")

    history_container = st.container()
    history_html = "<div style='height: 600px; overflow-y: auto; padding-right: 5px;'>"
    if st.session_state.qa_history:
        for item in st.session_state.qa_history:
            history_html += f"<b>Q:</b> {item['question']}<br>"
            history_html += f"<b>A:</b> {item['answer']}<br><hr>"
    else:
        history_html += "No questions yet."
    history_html += "</div>"
    history_container.markdown(history_html, unsafe_allow_html=True)

with col2:
    user_query = st.chat_input("Ask about NareshIT courses, trainers, batches...")

    if user_query:
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        # Translate to English
        translated_q = GoogleTranslator(source=lang_code, target="en").translate(user_query)

        # Run Agent
        response = agent.run(f"{system_instruction}\n\nQuestion: {translated_q}")

        # Translate back
        translated_a = GoogleTranslator(source="en", target=lang_code).translate(response)

        st.session_state.qa_history.append({
            "question": user_query,
            "answer": translated_a
        })

        placeholder.markdown(f"**Answer:** {translated_a}")

# --- Sticky Footer ---
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    text-align: center;
    background: #a0d2eb;
    color: white;
    padding: 10px 0;
    font-size: 13px;
    border-top-left-radius: 15px;
    border-top-right-radius: 15px;
    box-shadow: 0 -3px 5px rgba(0,0,0,0.2);
}
.footer a {
    color: #fff;
    text-decoration: underline;
    font-weight: bold;
    margin: 0 10px;
}
</style>

<div class="footer">
    Built with ❤️ using LangChain, Groq, Hugging Face, Chroma & Streamlit <br>
    <a href='https://nareshit.in/' target='_blank'>🌐 NareshIT Website</a> | 
    <a href='https://www.instagram.com/nareshitech/?hl=en' target='_blank'>📸 Instagram</a>
</div>
""", unsafe_allow_html=True)
