from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END, START

class State(TypedDict):
    user_input: str
    response: str
    input_type: Literal["question", "command", "unknown"]

workflow = StateGraph(State)

def analyze_input(state: State) -> State:
    user_input = state["user_input"].lower()
    
    if user_input.endswith("?"):
        return {**state, "input_type": "question"}
    elif user_input.startswith("!"):
        return {**state, "input_type": "command"}
    else:
        return {**state, "input_type": "unknown"}

def get_next_node(state: State) -> Literal["question", "command", "unknown"]:
    return state["input_type"]

def answer_question(state: State) -> State:
    return {
        "user_input": state["user_input"],
        "response": "질문을 입력해주세요",
        "input_type": state["input_type"]
    }

def execute_command(state: State) -> State:
    return {
        "user_input": state["user_input"],
        "response": "커맨드 실행 완료",
        "input_type": state["input_type"]
    }

def ask_clarification(state: State) -> State:
    return {
        "user_input": state["user_input"],
        "response": "명령어를 입력해주세요",
        "input_type": state["input_type"]
    }

# 모든 노드 먼저 추가
workflow.add_node("parse_input", analyze_input)
workflow.add_node("answer_question", answer_question)
workflow.add_node("execute_command", execute_command)
workflow.add_node("ask_clarification", ask_clarification)

# START에서 parse_input으로 가는 엣지 추가
workflow.add_edge(START, "parse_input")

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "parse_input",
    get_next_node,
    {
        "question": "answer_question",
        "command": "execute_command",
        "unknown": "ask_clarification"
    }
)

# 엣지 추가
workflow.add_edge("answer_question", END)
workflow.add_edge("execute_command", END)
workflow.add_edge("ask_clarification", "parse_input")

# 그래프 컴파일
app = workflow.compile()

# 유저 입력 받기
user_input = input("Enter your input: ")

# 그래프 실행
result = app.invoke({"user_input": user_input, "response": "", "input_type": "unknown"})
print(result["response"])  
