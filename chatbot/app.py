import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env file
load_dotenv()

# ---------- Page Setup ----------
st.set_page_config(page_title="Intelligent Assistant")

# ---------- A Little Custom CSS ----------
# Just a few small style tweaks — nothing fancy.
st.markdown("""
<style>
.block-container {
    max-width: 800px;
    padding-top: 2rem;
}

h1 {
    font-weight: 700;
}

.stChatInput {
    border-radius: 10px;
}

hr {
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("Intelligent Assistant")


# ---------- Build the Chain (only once, not every rerun) ----------
@st.cache_resource
def load_chain():
    # This tells the AI how to format its answers (using markdown headings).
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful AI assistant.
Always answer using this markdown format:

# Topic
Short introduction.

## Key Points
- Point 1
- Point 2
- Point 3

## Explanation
Explain clearly.

## Example
Give one example.

## Summary
Short conclusion.
"""),
        ("user", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    output_parser = StrOutputParser()

    # Connects: prompt -> model -> parser
    return prompt | llm | output_parser


chain = load_chain()

# ---------- Keep Chat History in Memory ----------
# session_state remembers data between reruns of the app.
if "history" not in st.session_state:
    st.session_state.history = []  # each item: {"question": ..., "answer": ...}

# ---------- Show Past Questions & Answers ----------
for item in st.session_state.history:
    st.markdown(f"**You:** {item['question']}")
    st.markdown(f"**Assistant:** {item['answer']}")
    st.divider()

# ---------- User Input ----------
question = st.chat_input("Ask a question...")

if question:
    st.markdown(f"**You:** {question}")

    with st.spinner("Generating response..."):
        try:
            answer = chain.invoke({"question": question})
            st.markdown(f"**Assistant:** {answer}")
            # Save this question & answer for next time the app reruns
            st.session_state.history.append({"question": question, "answer": answer})
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                st.error("Daily limit reached (20 requests/day) — please wait a bit and try again.")
            else:
                st.error("Something went wrong. Please try again.")
                st.exception(e)

# ---------- Sidebar Info ----------
with st.sidebar:
    st.header("About")
    st.write("""
    This app shows how to use **LangChain** with **Google Gemini**
    inside a simple **Streamlit** chat app.

    **Tech used:**
    - Python
    - LangChain
    - Gemini 2.5 Flash
    - Streamlit
    """)

    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# ---------- Footer ----------
st.divider()
st.caption("Powered by LangChain & Google Gemini")