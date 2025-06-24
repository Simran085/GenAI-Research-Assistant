# Smart Assistant for Research Summarization

An AI-powered research document assistant (PDF/TXT) with:

- Auto summary (≤150 words)
- **Ask Anything**(RAG Q&A grounded in your doc, with citation and memory)
- **Challenge Me** (auto-generated comprehension questions with feedback)
- Attractive, user-friendly Streamlit UI
- Powered by **Groq’s Llama 3** for blazing fast, accurate answers.

---

## 🚀 **Features**
- Upload research reports or papers in PDF or TXT.
- Get a concise, document-grounded summary instantly.
- Toggle between two modes:
    - **Ask Anything**: Ask free-form questions; receive justified, cited, in-context answers. Conversation memory supports follow-ups.
    - **Challenge Me**: Receive new, logic-based questions each time; answer and get evaluated with explanations.
- All answers and feedback are cited to your uploaded content, no hallucination.
- Chat history appears in the sidebar for context (no clutter).

---

## 🛠️ **Setup Instructions**
### 1. Clone the Repo and Prepare the Environment
```bash
   git clone https://github.com/Simran085/GenAI-Research-Assistant
   cd <your-project-folder>
   python -m venv venv

Choose one of the following depending on your OS:
   source venv/Scripts/activate  # Windows Git Bash
   venv\Scripts\activate         # Windows CMD
   source venv/bin/activate      # Mac/Linux
```
### 2. Install Required Packages
```bash
   pip install -r requirements.txt
```
### 3. Get a Groq API Key
   - Sign up (free) at [groq](https://groq.com/)
   - Go to the dashboard → API Keys → “Create API Key”
   - Copy your key (gsk_...)

### 4. Create a .env File
In the project root directory (same location as app.py), create a file named .env and paste:
```bash
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx
```
### 5. Run the App
- Windows CMD or Git Bash:
```bash
   venv\Scripts\python -m streamlit run app.py
```
- Mac/Linux:
```bash
   streamlit run app.py
```

## 🖥️**App Architecture & Reasoning Flow**
**1. Document Ingestion**
* User uploads a **PDF/TXT** file.
* Text is extracted, cleaned, and split into overlapping “chunks” for retrieval.

**2. Vector Embedding & Indexing**
- Each chunk is embedded using a local SentenceTransformer model.
- A FAISS index is built for semantic retrieval (fast “search” for relevant text).

**3. RAG Pipeline for Answers**
- User’s question is embedded and used to retrieve top-matching chunks.
- The app constructs a prompt: context chunks + (optionally) conversation history + the user’s question.
- This prompt is sent to **Groq’s Llama 3 API**, with instructions to answer only from the retrieved context, and cite the chunk or quote text.
- The answer and cited support are shown, with the supporting snippet highlighted.

**4. Conversation Memory**
- Recent Q&A pairs are stored in session.
- On follow-up questions, last 3 interactions are included in the LLM prompt so answers can refer to prior context.

**5. Challenge Me Mode**
- “Generate New Questions” always gives a fresh set of 3 logic-based, context-grounded questions.
- User submits answers, receives auto-evaluation and document-based feedback.

**6. User Interface**
- Streamlit UI with sidebar navigation for Home/Summary, Ask Anything, and Challenge Me modes.
- All major interactions and chat history are cleanly separated.
- Supporting text is only shown on demand; no clutter.


## 🤝 **License & Credits**
- Built with ❤️ using **Streamlit**, **Groq**, **LangChain**, **FAISS**, and **SentenceTransformers**.
- For research and educational use.

**Enjoy your Smart Assistant for Research Summarization!**
