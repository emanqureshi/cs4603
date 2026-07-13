"""
CS4603 study-assistant agent — Mosaic AI Agent Framework version.

This is the SAME agent as `15.databricks_deployment/agent.py` (same 4 tools,
same system prompt), rebuilt to use the modern Databricks Agent Framework so it
can be deployed with `databricks.agents.deploy()`.

Two things changed versus v1:

1. LLM client
   v1: `ChatOpenAI(base_url=HOST/serving-endpoints, api_key=DATABRICKS_TOKEN)`
       — needs DATABRICKS_HOST/TOKEN/MODEL injected as secrets.
   v2: `ChatDatabricks(endpoint=...)` — authentication is AUTOMATIC. When the
       model is logged with a `DatabricksServingEndpoint` resource (see the
       notebook), Model Serving mints a short-lived credential at runtime. No
       secret scope, no DATABRICKS_TOKEN.

2. Interface
   v1: a bare LangGraph `MessagesState` graph passed to `mlflow.langchain`.
   v2: the graph is wrapped in `LangGraphChatAgent(ChatAgent)` and built on
       `ChatAgentState` / `ChatAgentToolNode`, so the model speaks the standard
       agent signature that `agents.deploy()`, the review app, AI Playground,
       and Agent Evaluation all understand.

`mlflow.models.set_model(AGENT)` at the bottom is the models-from-code hook —
exactly the same idea as v1.
"""

from typing import Any, Generator, Optional

import mlflow
from databricks_langchain import ChatDatabricks
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

# ─── LLM ─────────────────────────────────────────────────────────────────────
# Same model as v1, but reached through ChatDatabricks. No token needed here:
# auth is provided automatically at serving time via the logged resource.
LLM_ENDPOINT_NAME = "databricks-qwen35-122b-a10b"
# `reasoning_effort="none"` disables the model's reasoning output. Without it,
# this Qwen reasoning model returns `content` as a LIST of blocks that includes
# a {"type": "reasoning", ...} part, which MLflow's ChatAgentMessage rejects
# (it only allows "text"/"image_url"/"input_audio"). Passing it via
# `extra_params` forwards it in the request body — the ChatDatabricks equivalent
# of v1's `ChatOpenAI(..., reasoning_effort="none")`.
llm = ChatDatabricks(
    endpoint=LLM_ENDPOINT_NAME,
    temperature=0,
    extra_params={"reasoning_effort": "none"},
)


# ─── Tools (identical behaviour to v1's agent.py) ────────────────────────────
@tool
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
        "power": lambda x, y: x**y,
    }
    if operation not in ops:
        return f"Unknown operation '{operation}'. Use: {', '.join(ops)}"
    result = ops[operation](a, b)
    return f"{a} {operation} {b} = {result}"


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: the temperature value to convert
        from_unit: source unit — 'C', 'F', or 'K'
        to_unit: target unit — 'C', 'F', or 'K'
    """
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return f"Unknown unit '{from_unit}'. Use C, F, or K."

    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return f"Unknown unit '{to_unit}'. Use C, F, or K."

    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"


@tool
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


@tool
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
    matches = [v for k, v in knowledge.items() if key in k or k in key]
    if matches:
        return matches[0]
    available = ", ".join(knowledge.keys())
    return f"Topic '{topic}' not found. Available topics: {available}"


tools = [calculate, convert_temperature, analyze_text, lookup_cs4603_topic]

SYSTEM_PROMPT = (
    "You are a helpful CS4603 study assistant. You can perform calculations, "
    "convert temperatures, analyze text, and look up course topics about LLMs "
    "and LangChain. Use the available tools when appropriate, and explain your "
    "answers clearly. When a user asks about a course topic, use the lookup tool "
    "first, then add your own explanation."
)


# ─── Build the graph on ChatAgentState ───────────────────────────────────────
# Same shape as v1 (agent ↔ tools loop), but the state and tool node come from
# mlflow so every message is already in the dict form the ChatAgent expects.
def create_tool_calling_agent(
    model: ChatDatabricks,
    tools: list,
    system_prompt: str,
) -> CompiledStateGraph:
    model = model.bind_tools(tools)

    def should_continue(state: ChatAgentState):
        last_message = state["messages"][-1]
        return "continue" if last_message.get("tool_calls") else "end"

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
    )
    model_runnable = preprocessor | model

    def call_model(state: ChatAgentState, config: RunnableConfig):
        response = model_runnable.invoke(state, config)
        # Reasoning models (databricks-qwen35-*) may return `content` as a LIST
        # of blocks including {"type": "reasoning", ...}. MLflow's ChatAgentState
        # reducer / ChatAgentToolNode convert this AIMessage into ChatAgent dict
        # form and reject the "reasoning" block. Flatten to the text now, at the
        # source, before it enters graph state. tool_calls are left untouched.
        if isinstance(response.content, list):
            response.content = "".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


# ─── Wrap as a ChatAgent ─────────────────────────────────────────────────────
class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    @staticmethod
    def _flatten_content(msg: dict) -> dict:
        """Coerce list-style message content into a plain string.

        Reasoning models (e.g. databricks-qwen35-*) can return `content` as a
        LIST of blocks like [{"type": "reasoning", ...}, {"type": "text", ...}].
        MLflow's ChatAgentMessage only accepts a string or text/image/audio
        parts, so it rejects the "reasoning" block. Here we keep the text parts
        and drop the rest, leaving `content` as a string.
        """
        content = msg.get("content")
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            return {**msg, "content": text}
        return msg

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        out: list[ChatAgentMessage] = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                out.extend(
                    ChatAgentMessage(**self._flatten_content(msg))
                    for msg in node_data["messages"]
                )
        return ChatAgentResponse(messages=out)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": self._flatten_content(msg)})
                    for msg in node_data["messages"]
                )


# Enable MLflow tracing for LangChain/LangGraph calls, then register the model.
mlflow.langchain.autolog()
agent_graph = create_tool_calling_agent(llm, tools, SYSTEM_PROMPT)
AGENT = LangGraphChatAgent(agent_graph)
mlflow.models.set_model(AGENT)
