from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import Annotated
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

import json
from langchain_core.messages import ToolMessage, HumanMessage


load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)


tool = TavilySearchResults(max_results=2)
tools = [tool]


llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State) -> State:
    """챗봇 노드 함수"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


class BasicToolNode:
    """마지막 AI 메세지에서 요청된 도구를 실행하는 노드"""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("인풋에 메세지가 없습니다.")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(ToolMessage(content=json.dumps(tool_result), name=tool_call["name"], tool_call_id=tool_call["id"]))
        return {"messages": outputs}


    
def route_tools(
    state: State,
):
    """
    마지막 메시지에 도구 호출이 있으면 ToolNode로 라우팅하기 위해 conditional_edge에서 사용한다.
    그렇지 않으면 종료로 라우팅한다.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools" # 도구를 사용해야 할 때 "tools"를 반환한다.
    return END



tool_node = BasicToolNode(tools)

# 노드 추가
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# 엣지 추가
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("tools", "chatbot")

# 조건부 엣지 추가
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}
)

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    messages = []
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            messages.append(value["messages"][-1].content)

    return "\n".join(messages)


user_input = input("Enter your input: ")
result = stream_graph_updates(user_input)
print(f"result: {result}")