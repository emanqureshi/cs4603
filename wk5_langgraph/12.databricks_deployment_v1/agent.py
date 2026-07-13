"""
LangGraph agent model definition for MLflow models-from-code logging.

This file defines the agent graph so MLflow can serialize it independently.
The agent is a CS4603 study assistant with math, text analysis, unit
conversion, and course knowledge tools.
"""

import os
import mlflow
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage


# ─── Tools ───────────────────────────────────────────────────────────────────

def calculate(a: float, b: float, operation: str) -> str:
    """Perform a math calculation on two numbers.

    Args:
        a: first number
        b: second number
        operation: one of 'add', 'subtract', 'multiply', 'divide', 'power'
    """
    ops = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: "Error: division by zero" if y == 0 else x / y,
        "power": lambda x, y: x ** y,
    }
    if operation not in ops:
        return f"Unknown operation '{operation}'. Use: {', '.join(ops)}"
    result = ops[operation](a, b)
    return f"{a} {operation} {b} = {result}"


def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: the temperature value to convert
        from_unit: source unit — 'C', 'F', or 'K'
        to_unit: target unit — 'C', 'F', or 'K'
    """
    # Normalize to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return f"Unknown unit '{from_unit}'. Use C, F, or K."

    # Convert from Celsius to target
    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return f"Unknown unit '{to_unit}'. Use C, F, or K."

    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"


def analyze_text(text: str) -> str:
    """Analyze text and return statistics: word count, sentence count,
    average word length, and estimated reading time.

    Args:
        text: the text to analyze
    """
    words = text.split()
    word_count = len(words)
    sentence_count = sum(1 for c in text if c in ".!?") or 1
    avg_word_len = sum(len(w.strip(".,!?;:")) for w in words) / max(word_count, 1)
    reading_time_sec = word_count / 3.5  # ~210 words per minute = 3.5 words/sec

    return (
        f"Words: {word_count} | "
        f"Sentences: {sentence_count} | "
        f"Avg word length: {avg_word_len:.1f} chars | "
        f"Reading time: {reading_time_sec:.0f}s"
    )


def lookup_cs4603_topic(topic: str) -> str:
    """Look up a CS4603 course topic and return a brief summary.
    Available topics cover LLM fundamentals and LangChain.

    Args:
        topic: keyword to search for, e.g. 'tokens', 'embeddings', 'RAG',
               'agents', 'langgraph', 'tool calling', 'prompt engineering'
    """
    knowledge = {
        "tokens": (
            "Tokens are sub-word units the LLM processes. A word like 'embedding' "
            "might be 1-3 tokens. Models have context windows measured in tokens "
            "(e.g. 128k). Use tiktoken to count tokens for OpenAI-compatible models."
        ),
        "embeddings": (
            "Embeddings map text to dense vectors (e.g. 1024-dim for "
            "databricks-gte-large-en). Similar texts have high cosine similarity. "
            "Used for semantic search, RAG retrieval, and clustering."
        ),
        "rag": (
            "Retrieval-Augmented Generation: chunk documents, embed them into a "
            "vector store, retrieve relevant chunks at query time, and feed them "
            "as context to the LLM. Reduces hallucination for domain-specific Q&A."
        ),
        "agents": (
            "Agents are LLMs that can decide which tools to call and in what order. "
            "The ReAct pattern: Reason → Act → Observe → repeat. LangGraph gives "
            "explicit control over the agent loop via a state graph."
        ),
        "langgraph": (
            "LangGraph models agent workflows as directed graphs. Nodes are "
            "functions, edges define control flow, and conditional edges enable "
            "routing (e.g. tool calls vs. final answer). Supports checkpointing, "
            "human-in-the-loop, and multi-agent patterns."
        ),
        "tool calling": (
            "Tool calling lets the LLM output structured function calls instead of "
            "plain text. The runtime executes the tool and returns results. Tools "
            "are defined as Python functions with typed args and docstrings."
        ),
        "prompt engineering": (
            "Techniques: system messages for persona/rules, few-shot examples, "
            "chain-of-thought prompting, structured output formats. Temperature "
            "controls randomness (0 = deterministic, 1 = creative)."
        ),
        "mlflow": (
            "MLflow tracks experiments, logs models, and manages the model "
            "lifecycle. mlflow.langchain supports LangGraph graphs natively. "
            "Models are registered in Unity Catalog and served via Databricks "
            "Model Serving endpoints."
        ),
    }

    key = topic.lower().strip()
    # Fuzzy match: check if the search term is contained in any key
    matches = [v for k, v in knowledge.items() if key in k or k in key]
    if matches:
        return matches[0]
    available = ", ".join(knowledge.keys())
    return f"Topic '{topic}' not found. Available topics: {available}"


# ─── Build graph ─────────────────────────────────────────────────────────────

tools = [calculate, convert_temperature, analyze_text, lookup_cs4603_topic]

SYSTEM_PROMPT = (
    "You are a helpful CS4603 study assistant. You can perform calculations, "
    "convert temperatures, analyze text, and look up course topics about LLMs "
    "and LangChain. Use the available tools when appropriate, and explain your "
    "answers clearly. When a user asks about a course topic, use the lookup tool "
    "first, then add your own explanation."
)

# Validate required environment variables before building the LLM client.
# If any are missing, raise a clear error so the serving container logs
# show exactly what went wrong (instead of a cryptic DEPLOYMENT_FAILED).
_token = os.environ.get("DATABRICKS_TOKEN")
_host = os.environ.get("DATABRICKS_HOST")
_model = os.environ.get("DATABRICKS_MODEL")

_missing = [name for name, val in [("DATABRICKS_TOKEN", _token), ("DATABRICKS_HOST", _host), ("DATABRICKS_MODEL", _model)] if not val]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(_missing)}. "
        "Ensure the secret scope (cs4603-deploy) is configured on the serving endpoint, "
        "or set these variables in your .env file for local testing."
    )

llm = ChatOpenAI(
    model=_model,
    api_key=_token,
    base_url=_host.rstrip("/") + "/serving-endpoints",
    reasoning_effort="none",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

mlflow.models.set_model(graph)
