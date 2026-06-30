"""
Human-in-the-loop (HITL) example: customer refund approval.

A small support agent handles a customer message. It has two tools:

  - lookup_order  -> safe, read-only, runs automatically
  - issue_refund  -> moves money, so it PAUSES for a human to approve

`HumanInTheLoopMiddleware` interrupts before `issue_refund` runs. The script
then asks a human (you, in the terminal) to approve, edit the amount, or reject,
and resumes the SAME thread with that decision.

Run:
    python 3.hitl_refund.py
"""

import os
import uuid

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


def build_llm() -> ChatOpenAI:
    """Build the Databricks-backed chat model from the repo's .env values."""
    load_dotenv()
    return ChatOpenAI(
        model=os.environ["DATABRICKS_MODEL"],
        api_key=os.environ["DATABRICKS_TOKEN"],
        base_url=f"{os.environ['DATABRICKS_HOST']}/serving-endpoints",
        reasoning_effort="none",
        temperature=0,
    )


# --- Fake order backend ---------------------------------------------------
ORDERS = {
    "A1001": {"item": "Wireless Headphones", "amount": 79.99, "status": "delivered"},
    "A1002": {"item": "USB-C Cable", "amount": 12.50, "status": "delivered"},
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID and return the item, amount, and status."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with id {order_id}."
    return f"Order {order_id}: {order['item']}, ${order['amount']:.2f}, status={order['status']}."


@tool
def issue_refund(order_id: str, amount: float, reason: str) -> str:
    """Issue a refund for an order. Needs the order id, refund amount, and a reason."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Cannot refund: no order found with id {order_id}."
    return f"Refund of ${amount:.2f} issued for order {order_id} (reason: {reason})."


def build_agent():
    """Create the agent with a HITL gate on the refund tool."""
    return create_agent(
        model=build_llm(),
        tools=[lookup_order, issue_refund],
        # A checkpointer is REQUIRED so the agent can pause and later resume.
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "lookup_order": False,  # read-only -> run automatically
                    "issue_refund": True,   # spends money -> require approval
                },
                description_prefix="Refund requires manager approval",
            ),
        ],
    )


# --- Helpers that work whether invoke returns a dict or a GraphOutput -----
def get_interrupts(result):
    if isinstance(result, dict):
        return result.get("__interrupt__")
    return getattr(result, "interrupts", None)


def get_messages(result):
    return result["messages"] if isinstance(result, dict) else result.value["messages"]


def ask_human(action_request: dict) -> dict:
    """Show the proposed tool call and turn the human's choice into a resume payload."""
    print("\n=== APPROVAL REQUIRED ===")
    print(f"Tool : {action_request['name']}")
    print(f"Args : {action_request['args']}")
    print("=========================")

    choice = input("Approve refund? [y]es / [n]o / [e]dit amount: ").strip().lower()

    if choice == "y":
        return {"decisions": [{"type": "approve"}]}

    if choice == "e":
        new_amount = float(input("Approved refund amount: ").strip())
        edited_args = dict(action_request["args"], amount=new_amount)
        return {
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {"name": action_request["name"], "args": edited_args},
                }
            ]
        }

    reason = input("Reason for rejection: ").strip() or "Refund not approved."
    return {"decisions": [{"type": "reject", "message": reason}]}


def handle_refund(agent):
    """Run one refund request through the agent, with a human approval step."""
    order_id = input("Order ID (e.g. A1001): ").strip()
    complaint = input("Customer complaint: ").strip() or "Customer requested a refund."

    # Each refund is its own conversation -> fresh thread id.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    user_request = (
        f"Customer complaint about order {order_id}: {complaint} "
        "Please look up the order and issue an appropriate refund."
    )

    # 1. Start the agent. It auto-runs lookup_order, then pauses before issue_refund.
    result = agent.invoke({"messages": [HumanMessage(content=user_request)]}, config=config)

    interrupts = get_interrupts(result)
    if not interrupts:
        # Nothing needed approval (e.g. order not found); the agent already finished.
        for m in get_messages(result):
            m.pretty_print()
        return

    # 2. Ask the human about the first pending action.
    action_request = interrupts[0].value["action_requests"][0]
    resume_payload = ask_human(action_request)

    # 3. Resume the SAME thread with the decision.
    result = agent.invoke(Command(resume=resume_payload), config=config)

    print("\n=== RESULT ===")
    for m in get_messages(result):
        m.pretty_print()


def main():
    agent = build_agent()

    while True:
        print("\n=== Refund Desk ===")
        print("1) Handle a refund request")
        print("0) Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            handle_refund(agent)
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid option, please try again.")


if __name__ == "__main__":
    main()
