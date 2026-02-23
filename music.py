import sqlite3
import pandas as pd
import time
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# ==================================================
# METRICS (MEASURABLE VALUES)
# ==================================================
SQL_CALLS = 0
AGENT_CALLS = 0

# ==================================================
# PERSONA (SYSTEM PROMPT)
# ==================================================
PERSONA = """
You are MuseBot, a Music Intelligence Assistant.

Rules:
- Use the SQL tool for all factual queries
- Never hallucinate songs or artists
- Always base answers on database results
- Present results clearly
"""

# ==================================================
# KNOWLEDGE (CSV → SQLITE)
# ==================================================
CSV_FILE = "Music_Dataset_Cleaned.csv"
DB_FILE = "music.db"

df = pd.read_csv(CSV_FILE)
df["track_name"] = df["track_name"].fillna("Unknown Track")
df["artist_name"] = df["artist_name"].fillna("Unknown Artist")
df["genre"] = df["genre"].str.lower().str.strip()

conn = sqlite3.connect(DB_FILE)
df.to_sql("tracks", conn, index=False, if_exists="replace")
conn.close()

# ==================================================
# MODEL
# ==================================================
llm = init_chat_model(
    model="qwen2.5:7b",
    model_provider="ollama",
    temperature=0.3
)

# ==================================================
# TOOL (SQL)
# ==================================================
@tool
def sql_query(query: str) -> List:
    """Execute SQL query on the music database."""
    global SQL_CALLS
    SQL_CALLS += 1

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows

# ==================================================
# LANGCHAIN AGENT (STABLE)
# ==================================================
agent_executor = initialize_agent(
    tools=[sql_query],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    system_message=PERSONA
)

# ==================================================
# STATE (LANGGRAPH)
# ==================================================
class MusicState(TypedDict):
    question: str
    messages: List
    answer: Optional[str]

# ==================================================
# ROUTER NODE
# ==================================================
def router_node(state: MusicState) -> MusicState:
    state["messages"].append(HumanMessage(content=state["question"]))
    return state

# ==================================================
# AGENT NODE (LANGCHAIN AGENT)
# ==================================================
def agent_node(state: MusicState) -> MusicState:
    global AGENT_CALLS
    AGENT_CALLS += 1

    response = agent_executor.run(state["question"])

    state["messages"].append(AIMessage(content=response))
    state["answer"] = response
    return state

# ==================================================
# STATEGRAPH
# ==================================================
graph = StateGraph(MusicState)

graph.add_node("router", router_node)
graph.add_node("agent", agent_node)

graph.add_edge(START, "router")
graph.add_edge("router", "agent")
graph.add_edge("agent", END)

app = graph.compile()

# ==================================================
# CHAT FUNCTION (RETURNS METRICS)
# ==================================================
def chat(question: str):
    global SQL_CALLS, AGENT_CALLS
    SQL_CALLS = 0
    AGENT_CALLS = 0

    start = time.perf_counter()

    result = app.invoke({
        "question": question,
        "messages": [],
        "answer": None
    })

    latency = time.perf_counter() - start

    return result["answer"], {
        "latency": latency,
        "sql_calls": SQL_CALLS,
        "agent_calls": AGENT_CALLS
    }

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    print("🎵 MuseBot — StateGraph + LangChain Agent")
    print("Type 'exit' to quit\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break

        answer, metrics = chat(q)
        print("\nMuseBot:", answer)
        print("Metrics:", metrics, "\n")
