# 🤖 Agentic AI Assistant for NareshIT  

An **Agentic RAG-based AI Assistant** built using **LangChain, Groq LLM, Chroma, and Streamlit** that automates NareshIT course-related queries, retrieves real-time batch information, and communicates intelligently with both students and trainers.  

---

## 🚀 Project Overview  

This project leverages **Retrieval-Augmented Generation (RAG)** and **Agentic AI workflows** to create a smart, multilingual assistant that:  
- Answers queries about **NareshIT courses, trainers, and schedules**.  
- Fetches **real-time batch data** through web scraping.  
- Automatically **emails course trainers** with details such as class links and schedules.  
- Supports **multiple languages** for broader accessibility.  

The assistant acts as a **dynamic information hub**, powered by **LLM reasoning, vector search**, and **LangChain agents**, enabling context-aware and goal-driven automation.  

---

## 🧩 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| **Frontend / UI** | Streamlit |
| **LLM** | Groq API (ChatGroq) |
| **Vector Database** | Chroma |
| **Embeddings** | Hugging Face (`all-MiniLM-L6-v2`) |
| **Framework** | LangChain |
| **Translation** | Deep Translator, Google Translator |
| **Web Scraping** | BeautifulSoup / Requests |
| **Email Automation** | Python SMTP / LangChain Tools |
| **Language** | Python 3.10+ |

---

## 🧠 Key Features  

- ⚙️ **Agentic Workflow** — Uses LangChain Agents to manage RAG retrieval, translation, and email tasks.  
- 🔍 **RAG Pipeline** — Combines vector-based retrieval with Groq LLM for context-aware, accurate answers.  
- 🌐 **Real-Time Web Scraping** — Fetches latest NareshIT course and batch details dynamically.  
- 📧 **Automated Email Communication** — Sends detailed class and trainer information directly to instructors.  
- 💬 **Multilingual Support** — Handles user input and output in multiple languages via translators.  
- 🧾 **Interactive Interface** — Streamlit UI for seamless user interactions.  

---

## 🧱 Project Architecture  

