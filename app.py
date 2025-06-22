import streamlit as st
import os
import fitz
import nltk
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from dotenv import load_dotenv

nltk.download('punkt')

# Utility functions remain unchanged
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode('utf-8')

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def get_top_k_chunks(question, model, index, chunks, k=3):
    q_embedding = model.encode([question])
    D, I = index.search(np.array(q_embedding), k)
    return [chunks[i] for i in I[0]], I[0]

def make_rag_prompt(question, chunks, indices, chat_history=None):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {indices[i]+1}]\n{chunk}\n\n"
    history_str = ""
    if chat_history:
        for idx, (prev_q, prev_a, _) in enumerate(chat_history):
            history_str += f"Previous Q{idx+1}: {prev_q}\nPrevious A{idx+1}: {prev_a}\n"
    prompt = (
        f"{history_str}"
        f"Using ONLY the context below, answer the user's question. "
        f"Clearly cite the chunk number(s) and quote the phrase that supports your answer, e.g., 'This is supported by Chunk 2: ...'. "
        f"If the answer is not present, reply: 'Not found in document.'\n\n"
        f"Context:\n{context}\n"
        f"Question: {question}\nAnswer:"
    )
    return prompt

def get_summary(llm, cleaned_text):
    context = cleaned_text[:3500]
    prompt = (
        f"Summarize the following document in no more than 150 words. "
        f"Keep only the main points.\n\nDocument:\n{context}\n\nSummary:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def make_challenge_prompt(chunks):
    context = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(chunks[:5])])
    prompt = (
        "Generate exactly 3 logic-based or comprehension-focused QUESTIONS about the following document. "
        "For each, provide the answer and a 1-sentence justification, all labeled like this:\n\n"
        "Question 1: ...\nAnswer 1: ...\nJustification 1: ...\n"
        "Question 2: ...\nAnswer 2: ...\nJustification 2: ...\n"
        "Question 3: ...\nAnswer 3: ...\nJustification 3: ...\n\n"
        "Context:\n" + context
    )
    return prompt

def check_answer_prompt(user_q, user_a, chunks, indices):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Chunk {indices[i]+1}]\n{chunk}\n\n"
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {user_q}\n"
        f"User's Answer: {user_a}\n"
        f"Evaluate the answer for correctness based only on the context above. "
        f"Reply with 'Correct' or 'Incorrect' and provide a one-sentence justification citing a relevant chunk."
    )
    return prompt

def highlight_support_snippet(answer, chunks, indices):
    cited_chunk_match = re.search(r'Chunk\s*(\d+)', answer)
    quote_match = re.search(r'“([^”]+)”|\'([^\']+)\'|"([^"]+)"', answer)
    support_snippet, chunk_number = None, None
    if cited_chunk_match:
        chunk_number = int(cited_chunk_match.group(1))
    if quote_match:
        quoted_text = next(g for g in quote_match.groups() if g)
        if chunk_number and 0 < chunk_number <= len(chunks):
            chunk_text = chunks[chunk_number-1]
            support_snippet = quoted_text if quoted_text in chunk_text else None
        else:
            for ch in chunks:
                if quoted_text in ch:
                    support_snippet = quoted_text
                    break
    return support_snippet, chunk_number

def sidebar_conversation(history):
    with st.sidebar:
        st.markdown("## 💬 Conversation History")
        for i, (q, a, support) in enumerate(history):
            st.markdown(f"<b>Q{i+1}:</b> {q}", unsafe_allow_html=True)
            st.markdown(f"<b>A{i+1}:</b> {a}", unsafe_allow_html=True)
            if support:
                st.markdown(
                    f"<span style='background:#f2ffe6'><b>Supporting:</b> {support}</span>",
                    unsafe_allow_html=True
                )
            st.markdown("---")

def main():
    load_dotenv()
    st.set_page_config(page_title="Smart Assistant for Research Summarization", layout="wide")
    st.title("🤖 Smart Assistant for Research Summarization")

    # Style
    st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #e8f0fe 0%, #e3ffe8 100%);}
        .stTextInput>div>div>input {border-radius:1.2rem;}
        .stButton>button {border-radius:1.2rem;background:#2563eb;color:white;font-weight:bold;}
        .st-bt {border-radius:1.5rem;}
        </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "challenge_questions" not in st.session_state:
        st.session_state["challenge_questions"] = []
    if "challenge_answers" not in st.session_state:
        st.session_state["challenge_answers"] = []
    if "challenge_just" not in st.session_state:
        st.session_state["challenge_just"] = []

    uploaded_file = st.file_uploader("📤 Upload PDF or TXT", type=["pdf", "txt"])
    
    # Navigation
    page = st.sidebar.radio(
        "🔀 Select Mode",
        ["Home / Summary", "Ask Anything", "Challenge Me"],
        index=0
    )

    # Data preparation and LLM setup
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = extract_text_from_txt(uploaded_file)
        cleaned_text = clean_text(raw_text)
        chunks = chunk_text(cleaned_text)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index, _ = build_faiss_index(chunks, model)
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(
            temperature=0.2,
            groq_api_key=groq_api_key,
            model_name="llama3-70b-8192"
        )
        # Only on Home/Summary page
        if page == "Home / Summary":
            st.markdown("### 📝 Document Summary")
            summary = get_summary(llm, cleaned_text)
            st.info(summary)
            st.markdown("""
                <hr>
                <div style='font-size:18px'>
                Welcome to <b>Smart Assistant for Research Summarization!</b><br>
                Select a mode from the left panel to start asking questions or challenge yourself.
                </div>
            """, unsafe_allow_html=True)

        # Ask Anything Mode Page
        if page == "Ask Anything":
            st.markdown("### 🧠 Ask Anything Mode")
            question = st.text_input("Type your question based on the uploaded document (follow-ups supported):", key="askanything")
            if question and st.button("Get Answer", key="getanswer"):
                with st.spinner("Getting answer from LLM..."):
                    chat_history = st.session_state["chat_history"][-3:] if st.session_state["chat_history"] else []
                    top_chunks, indices = get_top_k_chunks(question, model, index, chunks)
                    prompt = make_rag_prompt(question, top_chunks, indices, chat_history)
                    response = llm.invoke(prompt)
                    answer = response.content.strip()
                    support_snippet, chunk_number = highlight_support_snippet(answer, top_chunks, indices)
                    st.session_state["chat_history"].append((question, answer, support_snippet))
                    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
                    st.markdown("### 🤖 Answer")
                    st.write(answer)
                    if support_snippet:
                        st.markdown("**Supporting Snippet:**")
                        st.markdown(
                            f"<div style='background-color:#fff7e6;padding:8px;border-radius:5px'><b>{support_snippet}</b></div>",
                            unsafe_allow_html=True
                        )
            if st.session_state["chat_history"]:
                sidebar_conversation(st.session_state["chat_history"])
        
        # Challenge Me Mode Page
        if page == "Challenge Me":
            st.markdown("### 🎯 Challenge Me Mode")
            if st.button("🎲 Generate New Challenge Questions"):
                with st.spinner("Generating challenge questions..."):
                    prompt = make_challenge_prompt(chunks)
                    response = llm.invoke(prompt)
                    raw = response.content.strip()
                    q_re = re.compile(r'Question\s*\d+:(.*?)(?=Answer\s*\d+:)', re.DOTALL)
                    a_re = re.compile(r'Answer\s*\d+:(.*?)(?=Justification\s*\d+:)', re.DOTALL)
                    j_re = re.compile(r'Justification\s*\d+:(.*?)(?=Question\s*\d+:|$)', re.DOTALL)
                    questions = [q.strip() for q in q_re.findall(raw)]
                    answers = [a.strip() for a in a_re.findall(raw)]
                    justifications = [j.strip() for j in j_re.findall(raw)]
                    st.session_state["challenge_questions"] = questions[:3]
                    st.session_state["challenge_answers"] = answers[:3]
                    st.session_state["challenge_just"] = justifications[:3]
                    st.success("New questions generated! Try to answer them below.")

            if st.session_state["challenge_questions"] and len(st.session_state["challenge_questions"]) == 3:
                user_responses = []
                for idx, q in enumerate(st.session_state["challenge_questions"]):
                    user_answer = st.text_input(f"Your answer for Q{idx+1}: {q}", key=f"challenge_{idx}")
                    user_responses.append(user_answer)
                if st.button("Submit Answers", key="submit_challenge"):
                    st.subheader("🧾 Evaluation & Feedback")
                    for idx, (user_q, user_a) in enumerate(zip(st.session_state["challenge_questions"], user_responses)):
                        if user_a.strip():
                            top_chunks, indices = get_top_k_chunks(user_q, model, index, chunks)
                            prompt = check_answer_prompt(user_q, user_a, top_chunks, indices)
                            feedback = llm.invoke(prompt).content.strip()
                            st.markdown(f"**Q{idx+1}: {user_q}**")
                            st.markdown(f"Your answer: {user_a}")
                            st.markdown(f"Feedback: {feedback}")

    else:
        st.markdown("""
        <div style='padding:30px; font-size:20px; color:#333;'>
        <b>Welcome!</b> Please upload a PDF or TXT research document to get started.<br>
        <ul>
            <li>Auto-summary will be generated.</li>
            <li>Toggle between 'Ask Anything' and 'Challenge Me' modes.</li>
            <li>All answers are grounded in your document!</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
