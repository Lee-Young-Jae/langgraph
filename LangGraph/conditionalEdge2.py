from typing import Literal, Optional, TypedDict
from langgraph.graph import StateGraph, END, START


# 데이터 처리 함수 (모의 데이터)
def extract_location(query: str) -> Optional[str]:
  # 쿼리에서 "in" 다음에 오는 단어를 위치로 간주
  words = query.split()
  if "in" in words:
    index = words.index("in")
    if index + 1 < len(words):
      return words[index + 1]
  return None

def is_ambiguous(location: str) -> bool:
  # 위치가 3글자 미만이면 애매한 것으로 간주
  return len(location) < 3

def fetch_weather_data(location: str) -> str:
  # 간단한 모의 날씨 데이터
  weather_data = {
    "New York": "오늘은 흐림이고 10도입니다.",
    "London": "오늘은 비가 오고 15도입니다.",
    "Paris": "오늘은 맑고 20도입니다.",
    "Tokyo": "오늘은 비가 오고 15도입니다.",
    "Seoul": "오늘은 맑고 20도입니다.",
    "서울": "오늘은 맑고 20도입니다.",
  }
  return weather_data.get(location, "날씨 데이터를 찾을 수 없습니다.")

def generate_location_clarification(location: str) -> str:
  # 위치가 애매한 경우 확인 요청
  return f"확인할 위치가 '{location}' 이 맞나요?"

def generate_weather_response(location: str, weather_data: str) -> str:
  # 날씨 데이터를 기반으로 응답 생성
  return f"현재 {location}의 날씨는 {weather_data}."

# 상태 정의
class WeatherState(TypedDict):
  query: str
  location: Optional[str]
  weather_data: Optional[str]
  response: Optional[str]

# 그래프 정의
graph = StateGraph(WeatherState)

# 노드 정의
def parse_query(state: WeatherState) -> WeatherState:
  location = extract_location(state["query"])
  return {**state, "location": location}

def get_forecast(state: WeatherState) -> WeatherState:
  forecast = fetch_weather_data(state["location"])
  return {**state, "weather_data": forecast}

def clarify_location(state: WeatherState) -> WeatherState:
  clarification = generate_location_clarification(state["location"])
  return {**state, "response": clarification}

def generate_response(state: WeatherState) -> WeatherState:
  response = generate_weather_response(state["location"], state["weather_data"])
  return {**state, "response": response}


def check_location(state: WeatherState) -> Literal["valid", "invalid", "ambiguous"]:
    if not state["location"]:
        return "invalid"
    elif is_ambiguous(state["location"]):
        return "ambiguous"
    else:
        return "valid"
    

# 그래프 정의
graph.add_node("parse_query", parse_query)
graph.add_node("get_forecast", get_forecast)
graph.add_node("check_location", check_location)
graph.add_node("clarify_location", clarify_location)
graph.add_node("generate_response", generate_response)

# START에서 parse_query로 가는 엣지 추가
graph.add_edge(START, "parse_query")

# 조건부 엣지 정의
graph.add_conditional_edges(
    "parse_query",
    check_location,
    {"valid": "get_forecast", "invalid": "generate_response", "ambiguous": "clarify_location"}
)

# 추가 엣지 정의
graph.add_edge("get_forecast", "generate_response")
graph.add_edge("clarify_location", END)
graph.add_edge("generate_response", END)

# 그래프 컴파일
app = graph.compile()


user_input = input("Enter your input: ")

# 그래프 실행
result = app.invoke({"query": user_input, "location": None, "weather_data": None, "response": None})
print(result["response"])