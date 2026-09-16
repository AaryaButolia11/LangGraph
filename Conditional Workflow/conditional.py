import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# Make sure your .env contains:
# GROQ_API_KEY=your_groq_api_key


# ============================================================
# 2. CREATE EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 3. BUILD RAG RETRIEVER
# ============================================================

def build_retriever(pdf_path: str):
    """
    Loads a PDF, splits it into chunks,
    creates embeddings, stores them in FAISS,
    and returns a retriever.
    """

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split PDF into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    # Convert vector database into retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    return retriever


# Build separate retrievers for separate PDFs
academic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")


# ============================================================
# 4. CREATE LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4,
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# 5. DEFINE LANGGRAPH STATE
# ============================================================

class State(TypedDict):
    programme: str

    # add_messages automatically manages conversation messages
    messages: Annotated[list, add_messages]

    # Stores classifier result
    query_type: str

    # Stores retrieved PDF context
    retrieved_context: str


# ============================================================
# 6. CLASSIFIER NODE
# ============================================================

def classifier_node(state: State) -> dict:
    """
    Looks at the latest user message and classifies it
    into academic, fee, or general.
    """

    # Get latest user message
    last_message = state["messages"][-1].content

    prompt = f"""
Classify the following student query into exactly one category:

academic
fee
general

Rules:

Use "academic" for questions about:
- attendance
- exams
- grading
- credits
- promotion
- course structure
- summer training
- degree requirements
- subjects
- academic rules

Use "fee" for questions about:
- tuition
- payment
- refund
- late charges
- scholarships
- fees
- hostel fees
- any money-related college topic

Use "general" for:
- greetings
- casual conversation
- general questions
- anything unrelated to college academic rules or fees

Query:
{last_message}

Return ONLY one word:
academic, fee, or general
"""

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    # Normalize the model output
    if "academic" in category:
        category = "academic"

    elif "fee" in category:
        category = "fee"

    else:
        category = "general"

    return {
        "query_type": category
    }


# ============================================================
# 7. ACADEMIC RAG NODE
# ============================================================

def academic_rag_node(state: State) -> dict:
    """
    Retrieves relevant chunks from academic.pdf.
    """

    query = state["messages"][-1].content

    # Search academic PDF
    docs = academic_retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# 8. FEE RAG NODE
# ============================================================

def fee_rag_node(state: State) -> dict:
    """
    Retrieves relevant chunks from fee.pdf.
    """

    query = state["messages"][-1].content

    # Search fee PDF
    docs = fee_retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "retrieved_context": context
    }


# ============================================================
# 9. GENERAL NODE
# ============================================================

def general_node(state: State) -> dict:
    """
    For general queries, no PDF retrieval is required.
    """

    return {
        "retrieved_context": "NO_RETRIEVAL_NEEDED"
    }


# ============================================================
# 10. RESPONSE NODE
# ============================================================

def response_node(state: State) -> dict:
    """
    Generates the final answer using:
    - student programme
    - retrieved PDF context
    - user query
    """

    query = state["messages"][-1].content

    programme = state.get(
        "programme",
        "Unknown"
    )

    context = state.get(
        "retrieved_context",
        ""
    )

    # --------------------------------------------------------
    # General query
    # --------------------------------------------------------

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = f"""
You are a friendly college assistant.

You are talking to a {programme} student.

Answer the following question using your general knowledge.

Question:
{query}

Give a concise, friendly and useful answer.
"""

    # --------------------------------------------------------
    # RAG query
    # --------------------------------------------------------

    else:

        prompt = f"""
You are a college assistant helping a {programme} student.

Use the following context retrieved from official college
documents to answer the student's question.

Important instructions:

1. Answer using the provided context.
2. Do not invent college-specific information.
3. If the answer is not available in the context,
   clearly say that the provided documents do not contain
   enough information.
4. If multiple programmes are mentioned, use the information
   relevant to the student's programme: {programme}.
5. Give a clear and precise answer.

Student programme:
{programme}

Context:
{context}

Question:
{query}

Answer:
"""

    # Call LLM
    response = llm.invoke(prompt)

    # Add AI response to messages
    return {
        "messages": [
            ("ai", response.content.strip())
        ]
    }


# ============================================================
# 11. ROUTER FUNCTION
# ============================================================

def route_query(state: State):
    """
    Decides which node to execute after classification.
    """

    query_type = state["query_type"]

    if query_type == "academic":
        return "academic_rag"

    elif query_type == "fee":
        return "fee_rag"

    else:
        return "general"


# ============================================================
# 12. CREATE STATE GRAPH
# ============================================================

graph = StateGraph(State)


# ============================================================
# 13. ADD NODES
# ============================================================

graph.add_node(
    "classifier",
    classifier_node
)

graph.add_node(
    "academic_rag",
    academic_rag_node
)

graph.add_node(
    "fee_rag",
    fee_rag_node
)

graph.add_node(
    "general",
    general_node
)

graph.add_node(
    "response",
    response_node
)


# ============================================================
# 14. ADD EDGES
# ============================================================

# START → classifier
graph.add_edge(
    START,
    "classifier"
)


# classifier → one of three paths
graph.add_conditional_edges(
    "classifier",
    route_query
)


# RAG/general → response
graph.add_edge(
    "academic_rag",
    "response"
)

graph.add_edge(
    "fee_rag",
    "response"
)

graph.add_edge(
    "general",
    "response"
)


# response → END
graph.add_edge(
    "response",
    END
)


# ============================================================
# 15. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 16. PROGRAMME SELECTION
# ============================================================

print(
    "\nWelcome to the College Assistant!"
)

print(
    "You can ask questions about academics, "
    "fees, or general queries."
)

print(
    "\nPlease enter your programme:"
)

print("1. BCA")
print("2. BBA")
print("3. B.Com (H)")


choice = input(
    "\nEnter 1, 2 or 3: "
).strip()


programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)"
}


# Default to BCA if invalid choice
student_programme = programme_map.get(
    choice,
    "BCA"
)


print(
    f"\nGreat! You're set as a "
    f"{student_programme} student."
)

print(
    "\nAsk your question."
)

print(
    "Type 'exit' or 'quit' to stop.\n"
)


# ============================================================
# 17. CHAT LOOP
# ============================================================

while True:

    user_query = input("You: ").strip()

    # Exit condition
    if user_query.lower() in ["exit", "quit"]:
        print("Assistant: Goodbye!")
        break

    # Ignore empty input
    if not user_query:
        continue

    # Run LangGraph
    result = app.invoke(
        {
            "programme": student_programme,

            "messages": [
                ("human", user_query)
            ],

            # Initial values for state fields
            "query_type": "",
            "retrieved_context": ""
        }
    )

    # Get final AI response
    print(
        f"Assistant: "
        f"{result['messages'][-1].content}\n"
    )