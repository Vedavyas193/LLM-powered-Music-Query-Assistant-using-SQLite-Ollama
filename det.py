import sqlite3
import pandas as pd
import time
from langchain_ollama import ChatOllama

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
# DETERMINISTIC PIPELINE
# ========================
def deterministic_chat(question: str):
    global SQL_CALLS, LLM_CALLS
    SQL_CALLS = 0
    LLM_CALLS = 0

    start = time.perf_counter()
    q = question.lower()

    if "pop" in q:
        keyword = "pop"
    elif "rock" in q:
        keyword = "rock"
    else:
        return "Unsupported query", 0, 0, 0

    SQL_CALLS += 1
    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    cur.execute("""
        SELECT title, artist, popularity
        FROM songs
        WHERE LOWER(genre) LIKE ?
        ORDER BY popularity DESC
        LIMIT 5
    """, (f"%{keyword}%",))
    rows = cur.fetchall()
    conn.close()

    LLM_CALLS += 1
    response = llm.invoke(
        f"List these songs clearly:\n{rows}"
    )

    latency = time.perf_counter() - start
    return response.content, latency, SQL_CALLS, LLM_CALLS


# ========================
# RUN
# ========================
if __name__ == "__main__":
    answer, latency, sql_calls, llm_calls = deterministic_chat(
        "show popular pop songs"
    )

    print(answer)
    print("\nMetrics:")
    print("Latency:", latency)
    print("SQL calls:", sql_calls)
    print("LLM calls:", llm_calls)
