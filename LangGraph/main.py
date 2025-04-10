from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# .env 파일에서 환경 변수 로드
load_dotenv()

# GPT-4o-mini 설정
gpt4o_mini = ChatOpenAI(
  model_name="gpt-4o-mini",
  temperature=0.7, # 생성 결과의 다양성
  max_tokens=1500, # 생성 결과의 최대 토큰 수
)

# GPT-4o 설정
gpt4o = ChatOpenAI(
    model_name="gpt-4o",  # GPT-4o에 해당하는 모델명
    temperature=0.7,
    max_tokens=300,
)

# GPT-4o-mini 사용
response_mini = gpt4o_mini.invoke([HumanMessage(content="안녕하세요?")])
print(response_mini.content)
