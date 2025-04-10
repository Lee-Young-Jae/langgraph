from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)
    
llm = ChatOpenAI(model="gpt-3.5-turbo")

def chatbot(state: State) -> State:
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

app = graph_builder.compile()

def stream_graph_updates(user_input: str):
    messages = []
    for event in app.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            messages.append(value["messages"][-1].content)

    return "\n".join(messages)

question = input("Enter your input: ")
result = stream_graph_updates(question)
print(result)