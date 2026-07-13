# 16. Databricks Deployment v2 — Mosaic AI Agent Framework

This folder deploys the **same** CS4603 study-assistant agent as
[`15.databricks_deployment`](../15.databricks_deployment/), but using the
**modern Databricks Agent Framework** (`databricks-agents`) instead of the raw
Model Serving SDK.

Same agent, same tools, same result — a cleaner, more capable deployment path.

## Files

| File | Purpose |
|------|---------|
| `agent_chat.py` | The agent as an MLflow `ChatAgent` (models-from-code). Same 4 tools as v1. |
| `deployment_v2.ipynb` | Log → register → **`agents.deploy()`** → test. Run inside Databricks. |

## v1 vs v2 at a glance

| | v1 (`15.databricks_deployment`) | v2 (this folder) |
|---|---|---|
| Model file | `agent.py` — bare LangGraph `MessagesState` graph | `agent_chat.py` — wrapped as `ChatAgent` |
| LLM client | `ChatOpenAI(base_url=.../serving-endpoints, api_key=TOKEN)` | `ChatDatabricks(endpoint=...)` |
| Log API | `mlflow.langchain.log_model()` | `mlflow.pyfunc.log_model(resources=[...])` |
| Deploy API | `w.serving_endpoints.create(EndpointCoreConfigInput(...))` | `agents.deploy(model, version)` |
| **Authentication** | **Manual** — create `cs4603-deploy` secret scope, inject `DATABRICKS_HOST/TOKEN/MODEL` as `environment_vars` | **Automatic** — declared via `resources=[...]` at log time |
| You also get | Serving endpoint | Endpoint **+ review app + inference tables + evaluation + monitoring** |

Both paths use the identical backbone: **models-from-code → Unity Catalog →
Model Serving → OpenAI-compatible endpoint.** v2 just removes the manual auth
plumbing and adds the agent tooling.

## The key idea: automatic authentication

In v1, the agent needed a token to call the LLM, so you had to:

1. create a secret scope (`cs4603-deploy`),
2. put `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_MODEL` in it,
3. reference them in `environment_vars={"...": "{{secrets/cs4603-deploy/...}}"}`.

In v2, you **declare** the resources the agent uses when you log it:

```python
from mlflow.models.resources import DatabricksServingEndpoint

mlflow.pyfunc.log_model(
    name="agent",
    python_model="agent_chat.py",
    resources=[DatabricksServingEndpoint(endpoint_name="databricks-qwen35-122b-a10b")],
    ...
)
```

At deploy time, Databricks reads that list and mints a **short-lived,
least-privilege credential** automatically. No PAT, no secret scope, nothing to
rotate. If your agent also used a Vector Search index or a UC function, you'd
add `DatabricksVectorSearchIndex(...)` / `DatabricksFunction(...)` to the same
list.

## How to run

1. Upload this folder (`agent_chat.py` + `deployment_v2.ipynb`) to your
   Databricks workspace.
2. Open `deployment_v2.ipynb` in Databricks and run the cells top to bottom.
3. Wait 3–8 minutes for the endpoint to reach **READY**, then run the test cell.

The endpoint is OpenAI-compatible — call it with `openai.OpenAI` exactly like any
other model:

```python
response = client.chat.completions.create(
    model=deployment.endpoint_name,
    messages=[{"role": "user", "content": "Convert 100 F to Celsius"}],
)
print(response.choices[0].message.content)  # not response[0].messages[-1]
```

## When to use which

- **v1 (raw SDK)** — good for understanding exactly what a serving endpoint is:
  you wire the config and credentials yourself.
- **v2 (Agent Framework)** — the recommended path for real agents. Less
  boilerplate, automatic auth, and you get evaluation/monitoring/review-app for
  free.
