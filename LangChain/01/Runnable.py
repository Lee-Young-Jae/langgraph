from dotenv import load_dotenv

load_dotenv()

# 데이터를 효과적으로 전달하는 방법
# RunnablePassthrough 는 입력을 변경하지 않거나 추가 키를 더하여 전달할 수 있습니다.
# RunnablePassthrough() 가 단독으로 호출되면, 단순히 입력을 받아 그대로 전달합니다.
# RunnablePassthrough.assign(...) 방식으로 호출되면, 입력을 받아 assign 함수에 전달된 추가 인수를 추가합니다.

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# prompt와 llm 생성

prompt = PromptTemplate.from_template("""
{input} 의 10 배는?
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# # 체인 생성
# chain = prompt | llm

# # 1개의 변수만 템플릿에 포함하고 있다면, 값만 전달하는 것도 가능
# result = chain.invoke(10)

# print(result)

# ------------------------------------------------------------

# RunnablePassthrough
# RunnablePassthrough는 runnable 객체,

from langchain_core.runnables import RunnablePassthrough

# RunnablePassthrough().invoke(10)

# runnable_chain = {"input": RunnablePassthrough()} | prompt | llm

# # dict 값이 RunnalbePassthrough() 로 변경됨

# result = runnable_chain.invoke({"input": 10})
# print(result)

# ------------------------------------------------------------

# RunnablePassthrough().invoke({"input": 10})
# (RunnablePassthrough.assign(new_input=lambda x: x["input"] * 3)).invoke({"input": 1})



# ------------------------------------------------------------

# ChatPromptTemplate
# 대화목록을 프롬프트로 주입하고자 할 때 활용할 수 있다.
# 메세지는 튜플 형식으로 구성하ㅕ, (role, message) 로 구성하여 리스트로 생성 할 수 있다.

# role
# - "system": 시스템 설정 메세지로, 주로 전역설정과 관련된 프롬프트 이다.
# - "human": 사용자 입력 메세지
# - "ai": 모델의 출력 메세지


from langchain_core.prompts import ChatPromptTemplate

# chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")

# print(chat_prompt.format(country="한국"))

# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "당신은 친절한 AI 어시스턴트로. 당신의 이름은 {name} 입니다."),
#     ("human", "반가워요!"),
#     ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
#     ("human", "{user_input}"),
# ])

# # 챗 message 를 생성합니다.
# messages = chat_template.format_messages(
#     name="자비스", 
#     user_input="당신의 이름은 무엇입니까?"
# )

# result = llm.invoke(messages)

# print(result)

# ------------------------------------------------------------


chat_template = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 AI 어시스턴트로. 당신의 이름은 {name} 입니다."),
    ("human", "반가워요!"),
    ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
    ("human", "{user_input}"),
])

chain = chat_template | llm

print(chain.invoke({"name": "Javis", "user_input": "당신의 이름은 무엇입니까?"}))

# ------------------------------------------------------------



