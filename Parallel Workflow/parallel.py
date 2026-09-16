
# TypedDict → used to define the structure of our LangGraph state
# Annotated → used to attach a reducer function to a state field

from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# StateGraph → creates the graph
# START → starting point of graph
# END → ending point of graph
from langgraph.graph import StateGraph, START, END


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Create LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)


# --------------------------------------------------
# 3. Reducer Function
# --------------------------------------------------

def merge_score_dicts(existing: dict, newupdate: dict) -> dict:

    # If there is no existing value,
    # simply return the new update
    if existing is None:
        return newupdate

    # Merge old dictionary + new dictionary
    #
    # Example:
    #
    # existing  = {"toxicity_level": 80}
    # newupdate = {"copyright_risk": 70}
    #
    # result:
    #
    # {
    #     "toxicity_level": 80,
    #     "copyright_risk": 70
    # }

    return {
        **existing,
        **newupdate
    }


# --------------------------------------------------
# 4. Define Graph State
# --------------------------------------------------

class AnalyzerState(TypedDict):

    # Original text that every branch will analyze
    raw_text: str

    # All branches will update this same key.
    #
    # Annotated tells LangGraph:
    #
    # "When multiple nodes update safety_scores,
    # use merge_score_dicts to combine them."

    safety_scores: Annotated[
        dict[str, int],
        merge_score_dicts
    ]


# --------------------------------------------------
# 5. Toxicity Node
# --------------------------------------------------

def toxicity_node(state: AnalyzerState) -> dict:

    print("\n[Branch 1] Analyzing Toxicity...")

    # Ask LLM to analyze toxicity
    prompt = (
        "Analyze this text for profanity, aggression, hate speech, "
        "or toxicity. Give a score from 0 to 100. "
        "Return ONLY the integer.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    # Convert LLM response into integer
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Return an update to safety_scores
    return {
        "safety_scores": {
            "toxicity_level": score
        }
    }


# --------------------------------------------------
# 6. Copyright Node
# --------------------------------------------------

def copyright_node(state: AnalyzerState) -> dict:

    print("\n[Branch 2] Analyzing Copyright...")

    prompt = (
        "Analyze this text for plagiarism, unoriginal content, "
        "or trademark risk. Give a score from 0 to 100. "
        "Return ONLY the integer.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # This also updates safety_scores.
    # The reducer will merge this with the other branches.

    return {
        "safety_scores": {
            "copyright_risk": score
        }
    }


# --------------------------------------------------
# 7. Culture Node
# --------------------------------------------------

def culture_node(state: AnalyzerState) -> dict:

    print("\n[Branch 3] Analyzing Cultural Sensitivity...")

    prompt = (
        "Analyze this text for cultural insensitivity, "
        "regional sensitivities, or offensive content. "
        "Give a score from 0 to 100. "
        "Return ONLY the integer.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0

    # Again, update the SAME safety_scores key.

    return {
        "safety_scores": {
            "cultural_insensitivity": score
        }
    }


# --------------------------------------------------
# 8. Create Graph
# --------------------------------------------------

builder = StateGraph(AnalyzerState)


# --------------------------------------------------
# 9. Add Nodes
# --------------------------------------------------

builder.add_node("toxicity_node", toxicity_node)
builder.add_node("copyright_node", copyright_node)
builder.add_node("culture_node", culture_node)


# --------------------------------------------------
# 10. Create Parallel Edges
# --------------------------------------------------

# All three nodes start directly from START.
#
# Therefore they can execute independently.

builder.add_edge(START, "toxicity_node")
builder.add_edge(START, "copyright_node")
builder.add_edge(START, "culture_node")


# Each branch ends after completing its task.

builder.add_edge("toxicity_node", END)
builder.add_edge("copyright_node", END)
builder.add_edge("culture_node", END)


# --------------------------------------------------
# 11. Compile Graph
# --------------------------------------------------

app = builder.compile()


# --------------------------------------------------
# 12. Input
# --------------------------------------------------

sample_script = """
Yo guys! Welcome back to the stream.

Today I am going to show you how to hack into
your friend's system using a script I copied
directly from an online forum.

Traditional security protocols are garbage
and anyone still using them is an idiot.
"""


# Initial state

initial_state = {
    "raw_text": sample_script,

    # Initially there are no scores.
    "safety_scores": {}
}


# --------------------------------------------------
# 13. Run Graph
# --------------------------------------------------

final_state = app.invoke(initial_state)


# --------------------------------------------------
# 14. Print Result
# --------------------------------------------------

print("\n--- FINAL SAFETY SCORES ---")

print(final_state["safety_scores"])
