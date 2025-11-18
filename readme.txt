Text2SQL Agent
==============

This project hosts a LangChain 1.0 `create_agent` + DeepSeek powered text-to-SQL agent that enforces
strict read-only policies, validates SQL, executes queries through a LangChain tool
backed by SQLAlchemy (PyMySQL), and finally responds with natural-language insights
while recording every step with loguru.

Quick Start
-----------
1. Create a virtual environment and install the package locally:
   ```
   pip install -e .
   ```
2. Copy `.env.example` (or edit `.env`) and provide at least:
   - `DEEPSEEK_API_KEY` (optionally `DEEPSEEK_API_BASE`, `DEEPSEEK_MODEL_NAME`)
   - Either `DB_URL` or `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
   - Set `SCHEMA_SOURCE=database` (default) to auto-read schema via
     `INFORMATION_SCHEMA`, or keep `SCHEMA_SOURCE=file` to load JSON.
   - Optional: `SCHEMA_INCLUDE_TABLES` to whitelist certain tables,
     `SCHEMA_FILE`, `TEXT2SQL_DEFAULT_LIMIT`, `TEXT2SQL_LOG_PATH`
3. If you use the JSON path, describe the available objects in `schema.json`
   (or point `SCHEMA_FILE` elsewhere). Each table entry lists the allowed
   columns; the LLM prompt and SQL validator both rely on this file.
4. Launch the agent through LangGraph:
   ```
   langgraph dev text2sql
   ```
   or import `agent` from `Nl2Sql.agent` and call
   `agent.invoke({"messages": [{"role": "user", "content": "..."}]})`.

FastAPI API
-----------
Need an HTTP surface for the agent? Run the bundled FastAPI app:
```
uvicorn Nl2Sql.api:app --host 0.0.0.0 --port 8000
```
- `GET /healthz` returns `{"status": "ok"}` for readiness probes.
- `POST /nl2sql/query` accepts JSON payloads such as:
  ```json
  {
    "question": "Which of the top 10 customers by revenue are located in Zhejiang?",
    "user_name": "web_user",
    "include_debug": true
  }
  ```
- Responses include the latest `final_answer`, any `todos` emitted by
  `TodoListMiddleware`, optional `tool_logs` and LangGraph `updates`, plus the
  `final_state` snapshot. Set `include_debug=false` to skip the verbose fields.

Execution Flow
--------------
1. The natural-language request enters the LangChain `create_agent` graph. The
   system prompt embeds schema metadata, read-only rules, and "always use tools"
   guidance so the model does not hallucinate unsupported tables.
2. The agent drafts SQL, inspects schemas (`db_list_tables`, `db_get_table_schema`,
   `db_preview_table`), validates plans via `db_explain_query`, and only then
   executes read-only SQL through the safe executor.
3. The tool returns JSON (`sql` + `rows`) which the agent summarizes in fluent
   Chinese before answering the user and echoing the SQL when appropriate.
4. If the question cannot be answered safely, the agent states the reason rather
   than fabricating output.

Schema Source
-------------
Set `SCHEMA_SOURCE=database` (default) to pull live metadata from your MySQL
instance using the same connection defined by `DB_URL`/`DB_HOST` etc. Optional
`SCHEMA_INCLUDE_TABLES` (comma-separated) limits the tables exposed to the
LLM/validator. To keep a static schema snapshot, change `SCHEMA_SOURCE=file`
and supply `SCHEMA_FILE` (e.g., `schema.json`).

Fallback Behavior
-----------------
If the user asks for something that cannot be answered by a read-only query,
the LLM is instructed to return:
```
SELECT 'Question cannot be answered with read-only SQL' AS message
```
The validator accepts this output and the executor simply returns the message
as a single-row result.
