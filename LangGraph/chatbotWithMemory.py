from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv() 

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# 도구가 호출될 때마다 다음 단계를 결정하기 위해 챗봇으로 돌아온다.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("\n사용자: ")
    if user_input.lower() in ['quit', 'exit', '종료']:
        print("대화를 종료합니다.")
        break
        
    events = graph.stream({"messages": [HumanMessage(content=user_input)]}, config=config)
    
    for event in events:
        if "chatbot" in event:
            for message in event["chatbot"]["messages"]:
                print(f"\n챗봇: {message.content}")