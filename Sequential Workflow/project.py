
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()


# -----------------------------
# 1. Define the State
# -----------------------------

class PipelineState(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# -----------------------------
# 2. Create LLM
# -----------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)


# -----------------------------
# 3. Define Nodes
# -----------------------------

def editor_node(state: PipelineState) -> dict:
    """
    Stage 1:
    Cleans grammar, spelling, and improves flow.
    """

    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the "
        "transition flow while keeping the core message intact. "
        "Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {
        "edited_text": response.content.strip()
    }


def scriptwriter_node(state: PipelineState) -> dict:
    """
    Stage 2:
    Converts edited text into an engaging video script.
    """

    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text "
        "and transform it into a highly engaging, punchy, conversational video "
        "script hook. Make it sound like a real person speaking passionately. "
        "Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "script_text": response.content.strip()
    }


def translator_node(state: PipelineState) -> dict:
    """
    Stage 3:
    Converts the script into natural Hinglish.
    """

    prompt = (
        "You are an expert content localizer for the Indian market. "
        "Take the following script and convert it into natural, flowing "
        "'Hinglish'. Do not simply translate it sentence-by-sentence. "
        "Use Hindi and English phrases naturally, just like an intellectual "
        "tech educator would speak on a live stream. Keep the energy high! "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {
        "final_output": response.content.strip()
    }


# -----------------------------
# 4. Create Graph
# -----------------------------

graph = StateGraph(PipelineState)


# -----------------------------
# 5. Add Nodes
# -----------------------------

graph.add_node("Editor Node", editor_node)
graph.add_node("Scriptwriter Node", scriptwriter_node)
graph.add_node("Translator Node", translator_node)


# -----------------------------
# 6. Add Edges
# -----------------------------

graph.add_edge(START, "Editor Node")
graph.add_edge("Editor Node", "Scriptwriter Node")
graph.add_edge("Scriptwriter Node", "Translator Node")
graph.add_edge("Translator Node", END)


# -----------------------------
# 7. Compile Graph
# -----------------------------

app = graph.compile()


# -----------------------------
# 8. Run Graph
# -----------------------------

result = app.invoke({
    "raw_input": (
        "AI agents are the future of tech. They can think, plan, "
        "and act on their own. LangGraph helps you build these "
        "agents with proper control and memory."
    )
})


# -----------------------------
# 9. Print Final Output
# -----------------------------

print("\n--- Final Output ---")
print(result["final_output"])

