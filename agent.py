import sqlite3
import pandas as pd
import time
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# ========================
# LOAD DATA INTO SQLITE
# ========================
df = pd.read_csv("songs_2000_2020_50k.csv")
conn = sqlite3.connect("songs.db")
df.to_sql("songs", conn, if_exists="replace", index=False)
conn.close()

# ========================
# MODEL (OLLAMA)
# ========================
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# ========================
# METRICS
# ========================
SQL_CALLS = 0
LLM_CALLS = 0

# ========================
# SQL TOOL
# ========================
@tool
def sql_query(query: str):
    """Run SQL query on the songs database."""
    global SQL_CALLS
    SQL_CALLS += 1

    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows

# ========================
# AGENT-STYLE PIPELINE
# ========================
def agent_chat(question: str):
    global SQL_CALLS, LLM_CALLS
    SQL_CALLS = 0
    LLM_CALLS = 0

    start = time.perf_counter()

    # 1️⃣ LLM decides SQL
    LLM_CALLS += 1
    sql_plan = llm.invoke(
        f"""
You are generating SQLite SQL.

User question: {question}

Rules:
- Table: songs
- Columns: title, artist, genre, popularity
- Genre values are not exact
- Use: LOWER(genre) LIKE '%keyword%'
- Do NOT use markdown
- Return ONLY SQL
"""
    ).content

    # Clean markdown if present
    sql_plan = sql_plan.replace("```sql", "").replace("```", "").strip()

    # 2️⃣ Execute SQL via tool
    rows = sql_query.invoke(sql_plan)

    # 3️⃣ LLM explains results
    LLM_CALLS += 1
    answer = llm.invoke(
    f"""
You are a data reporting assistant.

Each tuple represents:
(title, artist, popularity_score)

Rules:
- Do NOT infer themes or lyrics
- Do NOT speculate
- Do NOT add context outside the data
- ONLY restate facts

Songs:
{rows}

Output format:
• Song title – Artist (Popularity score)
"""
)



    latency = time.perf_counter() - start
    return answer.content, latency, SQL_CALLS, LLM_CALLS


# ========================
# RUN
# ========================
if __name__ == "__main__":
    answer, latency, sql_calls, llm_calls = agent_chat(
        "show popular pop songs"
    )

    print(answer)
    print("\nMetrics:")
    print("Latency:", latency)
    print("SQL calls:", sql_calls)
    print("LLM calls:", llm_calls)
