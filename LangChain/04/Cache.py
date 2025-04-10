# LangChain은 선택적 캐싱 레이어를 제공한다.

# 동일한 완료를 여러 번 요청 하는 경우 LLM 공급자에 대한 API 호출 횟수를 줄임
# LLM 제공업체에 대한 API 호출횟수를 줄여 비용과, 속도를 향상

from dotenv import load_dotenv
import timeit
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def run_without_cache():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = PromptTemplate.from_template("{country} 에 대해 30자 내외로 요약해줘")
    chain = prompt | llm
    return chain.invoke({"country": "한국"})

def run_with_cache():
    set_llm_cache(InMemoryCache())
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    prompt = PromptTemplate.from_template("{country} 에 대해 30자 내외로 요약해줘")
    chain = prompt | llm
    return chain.invoke({"country": "한국"})

# 캐시 없이 실행
print("캐시 없이 실행:")
no_cache_time = timeit.timeit(run_without_cache, number=1)
print(f"실행 시간: {no_cache_time:.2f}초")

# 캐시와 함께 실행
print("\n캐시와 함께 실행:")
with_cache_time = timeit.timeit(run_with_cache, number=1)
print(f"실행 시간: {with_cache_time:.2f}초")

# 두 번째 실행 (캐시된 결과 사용)
print("\n캐시된 결과로 두 번째 실행:")
second_run_time = timeit.timeit(run_with_cache, number=1)
print(f"실행 시간: {second_run_time:.2f}초")