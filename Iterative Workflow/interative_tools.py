import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. TOOLS
# ============================================================

search_tool = TavilySearch(
    max_results=3,
    api_key=os.environ.get("TAVILY_API_KEY")
)

tools = [search_tool]


# ============================================================
# 3. LLMs
# ============================================================

# Writer LLM
writer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

# Writer can use tools
writer_llm_with_tools = writer_llm.bind_tools(tools)


# Reviewer LLM
reviewer_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.1
)


# ============================================================
# 4. STATE
# ============================================================

class State(TypedDict):
    topic: str

    # add_messages appends new messages instead of replacing them
    messages: Annotated[list, add_messages]

    draft: str
    review_feedback: str
    is_approved: bool
    attempt: int


# ============================================================
# 5. WRITER SYSTEM PROMPT
# ============================================================

WRITER_SYSTEM_PROMPT = """
You are an expert LinkedIn content writer.

Your job is to write engaging and professional LinkedIn posts.

If the topic requires:
- up-to-date information
- current statistics
- recent events
- current trends

use the web search tool before writing.

If you receive feedback on a previous draft, carefully address
every issue mentioned in the feedback.

Rules for a good LinkedIn post:

1. Strong hook in the first line.
2. One clear takeaway.
3. Easy to skim using short paragraphs.
4. Around 150-200 words.
5. End with a question or call-to-action.
6. Professional but human tone.
7. Do not use hashtags.
"""


# ============================================================
# 6. WRITER NODE
# ============================================================

def writer_node(state: State) -> dict:
    """
    Creates the first draft or rewrites the draft
    using reviewer feedback.
    """

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    previous_feedback = state.get(
        "review_feedback",
        ""
    )

    # First attempt
    if attempt == 1:

        user_message = (
            f"Write a LinkedIn post about: {topic}\n\n"
            "If current information is needed, search the web first."
        )

    # Revision attempt
    else:

        user_message = (
            f"Your previous LinkedIn post about '{topic}' "
            f"was rejected.\n\n"
            f"Reviewer's feedback:\n"
            f"{previous_feedback}\n\n"
            "Write a new improved draft that fixes every issue "
            "mentioned in the feedback. Do not repeat the same mistakes."
        )

    messages = [
        ("system", WRITER_SYSTEM_PROMPT),
        ("human", user_message)
    ]

    response = writer_llm_with_tools.invoke(messages)

    return {
        "messages": [
            ("human", user_message),
            response
        ],
        "attempt": attempt
    }


# ============================================================
# 7. TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 8. EXTRACT DRAFT NODE
# ============================================================

def extract_draft_node(state: State) -> dict:
    """
    Takes the final writer message and stores its text
    as the current draft.
    """

    last_message = state["messages"][-1]

    draft = last_message.content

    print("\nGenerated post:\n")
    print(draft)

    return {
        "draft": draft
    }


# ============================================================
# 9. REVIEWER SYSTEM PROMPT
# ============================================================

REVIEWER_SYSTEM_PROMPT = """
You are a strict LinkedIn content reviewer.

Judge whether a post is publish-ready.

Evaluate these criteria:

1. Strong hook in the first line.
2. One clear and valuable takeaway.
3. Easy to skim using short paragraphs.
4. Roughly 150-200 words.
5. Ends with an engaging question or CTA.
6. Professional but human tone.
7. No hashtags.

Respond in exactly this format:

VERDICT: APPROVED or REJECTED
FEEDBACK: <one short paragraph explaining why>

Be strict but fair.

Approve only if the post genuinely meets all criteria.
Reject if even one criterion is clearly missing.
"""


# ============================================================
# 10. REVIEWER NODE
# ============================================================

def reviewer_node(state: State) -> dict:
    """
    Reviews the current draft and decides whether
    it should be approved or rewritten.
    """

    draft = state["draft"]

    prompt = (
        "Review this LinkedIn post draft:\n\n"
        f"{draft}\n\n"
        "Give your review using the required format."
    )

    response = reviewer_llm.invoke(
        [
            ("system", REVIEWER_SYSTEM_PROMPT),
            ("human", prompt)
        ]
    )

    review_text = response.content.strip()

    # --------------------------------------------------------
    # Extract verdict
    # --------------------------------------------------------

    first_part = review_text.upper().split(
        "FEEDBACK",
        1
    )[0]

    is_approved = "APPROVED" in first_part

    # --------------------------------------------------------
    # Extract feedback
    # --------------------------------------------------------

    if "FEEDBACK:" in review_text:

        feedback = review_text.split(
            "FEEDBACK:",
            1
        )[1].strip()

    else:

        feedback = review_text

    verdict = (
        "APPROVED"
        if is_approved
        else "REJECTED"
    )

    print(f"\n[Verdict: {verdict}]")
    print(f"[Feedback: {feedback}]\n")

    return {
        "review_feedback": feedback,
        "is_approved": is_approved
    }


# ============================================================
# 11. ROUTER: SHOULD WRITER USE TOOL?
# ============================================================

def should_use_tool(state: State):
    """
    After writer runs:
    - if it requested a tool → execute tools
    - otherwise → extract the draft
    """

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):

        return "tools"

    return "extract_draft"


# ============================================================
# 12. ROUTER: SHOULD CONTINUE LOOP?
# ============================================================

def should_stop_looping(state: State):
    """
    After reviewer:
    - approved → END
    - 3 attempts reached → END
    - otherwise → writer for revision
    """

    if state["is_approved"]:

        print("\nPost has been approved.")

        return END

    if state["attempt"] >= 3:

        print("\nReached maximum attempts.")

        return END

    return "writer"


# ============================================================
# 13. BUILD GRAPH
# ============================================================

graph = StateGraph(State)


# ============================================================
# 14. ADD NODES
# ============================================================

graph.add_node(
    "writer",
    writer_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_node(
    "extract_draft",
    extract_draft_node
)

graph.add_node(
    "reviewer",
    reviewer_node
)


# ============================================================
# 15. EDGES
# ============================================================

# START → WRITER
graph.add_edge(
    START,
    "writer"
)


# WRITER → TOOLS or EXTRACT_DRAFT
graph.add_conditional_edges(
    "writer",
    should_use_tool
)


# IMPORTANT:
# Tool result must go back to writer
# so the writer can see the search results.
graph.add_edge(
    "tools",
    "writer"
)


# Once writer finishes without requesting tools
# → extract the draft
graph.add_edge(
    "extract_draft",
    "reviewer"
)


# Reviewer → END or WRITER
graph.add_conditional_edges(
    "reviewer",
    should_stop_looping
)


# ============================================================
# 16. COMPILE GRAPH
# ============================================================

app = graph.compile()


# ============================================================
# 17. USER INPUT
# ============================================================

print("=" * 55)
print("WELCOME TO THE LINKEDIN POST GENERATOR")
print("=" * 55)

print(
    "\nThis tool will:"
    "\n1. Generate a LinkedIn post"
    "\n2. Search the web when needed"
    "\n3. Review the post"
    "\n4. Rewrite it if rejected"
    "\n5. Stop when approved or after 3 attempts"
)

print("=" * 55)


topic = input(
    "\nWhat topic do you want a LinkedIn post about?\n> "
).strip()


# ============================================================
# 18. RUN GRAPH
# ============================================================

if not topic:

    print("\nNo topic given. Exiting.")

else:

    print("\nStarting generation...\n")

    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0
    }

    final_state = app.invoke(
        initial_state
    )


    # ========================================================
    # 19. FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 55)
    print("FINAL LINKEDIN POST")
    print("=" * 55)

    print(final_state["draft"])

    print("=" * 55)

    print(
        f"Total attempts: "
        f"{final_state['attempt']}"
    )

    print(
        f"Approved: "
        f"{final_state['is_approved']}"
    )