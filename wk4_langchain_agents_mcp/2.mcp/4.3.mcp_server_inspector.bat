@REM Launch the MCP Inspector (a browser-based debugging UI for MCP servers).
@REM   npx @modelcontextprotocol/inspector  -> downloads/runs the Inspector tool via Node/npm
@REM   python ...4.mcp_server_basic.py       -> the command the Inspector spawns as the MCP server
@REM
@REM The Inspector launches the server as a child process and talks to it over STDIO
@REM (no --http flag), so this connects over the stdio transport, not HTTP. It opens a
@REM local web page where you can list/call the server's tools, resources, and prompts.
@REM Requires Node.js (>= 22.7.5, for npx) and the project's Python env on PATH.

npx @modelcontextprotocol/inspector@latest python C:\\Users\\akhawaja\\git\\cs4603\\wk4_langchain_agents_mcp\\4.mcp_server_basic.py