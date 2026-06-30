"""Command-line client for the Chinook stdio MCP server (4.4.mcp_server_db.py).

Run it from the terminal:

    python 4.5.mcp_client_db.py

It launches the server as a child process over stdio (no separate server
terminal needed) and shows a small menu so you can call the `search_tracks`
tool and render the `music_analyst` prompt.
"""

import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def _text_of(contents) -> str:
    """Join the text from a list of MCP content blocks."""
    parts = []
    for block in contents:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts)


async def do_search(session: ClientSession) -> None:
    query = input("Search for (track / album / artist): ").strip()
    if not query:
        print("Nothing to search for.\n")
        return

    result = await session.call_tool("search_tracks", {"query": query, "limit": 5})

    # FastMCP returns the dict under a "result" key in structuredContent.
    data = (result.structuredContent or {}).get("result", {})
    matches = data.get("results", [])
    print(f"\nFound {data.get('count', 0)} track(s) for {query!r}:")
    for m in matches:
        print(f"  - {m['track']}  |  {m['artist']}  |  {m['album']}  "
              f"|  {m['genre']}  |  {m['duration']}  |  ${m['price']}")
    if not matches:
        print("  (no matches)")
    print()


async def do_prompt(session: ClientSession) -> None:
    question = input("Ask the music analyst: ").strip()

    result = await session.get_prompt("music_analyst", {"question": question})
    print("\n--- Rendered prompt (what you'd send to an LLM) ---")
    for msg in result.messages:
        print(f"[{msg.role}]")
        print(_text_of([msg.content]))
    print("---------------------------------------------------\n")


async def do_resources(session: ClientSession) -> None:
    # List every resource the server advertises, let the user pick one,
    # then fetch and display it.
    listing = await session.list_resources()
    resources = listing.resources
    if not resources:
        print("The server exposes no resources.\n")
        return

    print("Available resources:")
    for i, res in enumerate(resources, start=1):
        label = res.name or str(res.uri)
        print(f"  {i}) {label}  [{res.uri}]")
    print()

    choice = input(f"Pick a resource [1-{len(resources)}] (or Enter to cancel): ").strip()
    if not choice:
        print("Cancelled.\n")
        return
    if not choice.isdigit() or not (1 <= int(choice) <= len(resources)):
        print("Invalid choice.\n")
        return

    selected = resources[int(choice) - 1]
    result = await session.read_resource(selected.uri)
    print(f"\n--- {selected.uri} ---")
    print(_text_of(result.contents))
    print("-" * (len(str(selected.uri)) + 8) + "\n")


MENU = """\
==== Chinook MCP client ====
  1) Search tracks        (call the search_tracks tool)
  2) Ask the music analyst (render the music_analyst prompt)
  3) View resources        (list resources, then read one)
  4) Quit
"""

SERVER_PATH = Path(__file__).parent / "4.4.mcp_server_db.py"

async def main() -> None:
    params = StdioServerParameters(command="python", args=[str(SERVER_PATH)])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to the Chinook MCP server.\n")

            actions = {"1": do_search, "2": do_prompt, "3": do_resources}
            while True:
                print(MENU)
                choice = input("Choose an option [1-4]: ").strip()
                print()
                if choice == "4":
                    print("Goodbye!")
                    return
                action = actions.get(choice)
                if action is None:
                    print("Invalid choice, please pick 1-4.\n")
                    continue
                try:
                    await action(session)
                except Exception as exc:  # keep the menu alive on any error
                    print(f"Error: {exc}\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye!")
