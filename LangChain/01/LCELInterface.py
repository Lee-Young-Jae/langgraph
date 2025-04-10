from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = PromptTemplate(
    template="""
    {text} 에 대하여 3줄 이하로 설명해줘.
    """
)

chain = prompt | model | StrOutputParser()



# 1.
# stream: 실시간 출력
# 이 함수는 `chain.stream` 메서드를 사용하여 주어진 토픽에 대한 데이터 스트림을 생성하고, 이 스트림을 반복하여 각 데이터의 내용을 즉시 출력한다.
# end="" 인자는 출력 후 줄바꿈을 하지 않도록 설정하며, 출력 속도를 높이기 위해 사용된다.
# `flush=True` 인자는 출력 후 즉시 버퍼를 비우도록 설정하여, 출력 속도를 높이기 위해 사용된다.

# user_input = input("Enter your input: ")


# for token in chain.stream({"text": user_input}):
#     print(token, end="", flush=True)


# 2.
# invoke: 호출
# `chain` 객체의 `invoke` 메서드는 주제를 인자로 받아 주제에 대한 처리 수행

# user_input = input("Enter your input: ")

# result = chain.invoke({"text": user_input})

# print(result)


# 3.
# bach: 배치(단위 실행)

# print("궁금한 토픽을 모두 입력하세요, 끝내라면 `끝` 혹은 `q`를 입력하세요.")

# topics = []

# while True:
#     user_input = input("Enter your input: ")
#     if user_input in ["끝", "q", "Q", "end", "END", "종료", "종료하기", ""]:
#         break
#     topics.append({"text": user_input})


# results = chain.batch(topics, config={"max_concurrency": 3})

# for result in results:
#     print(result)


# 4.
# async stream: 비동기 스트림
# 함수 `chain.astream은 비동기 스트림을 생성하며, 주어진 토픽에 대한 메시지를 비동기적으로 처리한다.

# 비동기 for 루프(async for)를 사용하여 스트림에서 메시지를 순차적으로 받아오고, `print` 를 통해 메시지의 내용
# (s.content)을 즉시 출력한다.
# `end=""` 인자는 출력 후 줄바꿈을 하지 않도록 설정하며, `flush=True` 인자는 출력 후 즉시 버퍼를 비우도록 한다.

# print("궁금한 토픽을 모두 입력하세요, 끝내라면 `끝` 혹은 `q`를 입력하세요.")

# topics = []

# while True:
#     user_input = input("Enter your input: ")
#     if user_input in ["끝", "q", "Q", "end", "END", "종료", "종료하기", ""]:
#         break
#     topics.append({"text": user_input})

# async for s in chain.astream(topics):
#     print(s.content, end="", flush=True)


# 이외에도 async invoke, async batch 등 비동기 처리 방식도 있음


# Parallel: 병렬성

# langchain_core.runnables 모듈의 RunnableParallel 클래스를 사용하여 두 가지 작업을 병렬로 실행하는 예시

# `ChatPromptTemplate.from_template` 메서드를 사용하여 country 에 대한 수도와 면적을 구하는 두개의 체인을 만든다.

# 이 체인들은 각각 model과 파이프 | 연산자를 통해 연결되며, 마지막으로 RunnableParallel 클래스를 사용하여 이 두 체인을 capital와 area 라는 키로 결합하여
# 동시에 실행할 수 있는 combined 객체를 생성한다.

from langchain_core.runnables import RunnableParallel

# {contry} 의 수도를 물어보는 체인 생성
chain1 = (
  PromptTemplate.from_template("{country}의 수도는 어디야?")
  | model
  | StrOutputParser()
)

# {country} 의 면적을 물어보는 체인 생성
chain2 = (
  PromptTemplate.from_template("{country}의 면적은 얼마야?")
  | model
  | StrOutputParser()
)

# # RunnableParallel 클래스를 사용하여 두 체인을 결합하여 병렬로 실행
combined = RunnableParallel({"capital": chain1, "area": chain2})


# # 두 체인을 병렬로 실행하여 결과를 출력
# result = combined.invoke({"country": "대한민국"})

# print(result)


# 배치에서의 병렬 처리

chain1.batch([{
  "country": "대한민국"
}, {
  "country": "미국"
}])

chain2.batch([{
  "country": "대한민국"
}, {
  "country": "미국"
}])

result = combined.batch([{
  "country": "대한민국"
}, {
  "country": "미국"
}])




print(result)

