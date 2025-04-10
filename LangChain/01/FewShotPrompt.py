# Few-shot 학습은 모델에게 한정된 수의 예시(샘플)를 제공하여 새로운 작업을 수행하도록 유도하는 학습 방식입니다.
# 전통적인 머신러닝 방식에서 "zero-shot" (예시 없이 바로 수행)과 비교할 때, few-shot은 몇 개의 예시를 통해 모델이 특정 작업의 맥락이나 요구사항을 파악하도록 돕습니다.

from typing import Generator, Any

# 스트림 응답 출력하는 함수

def stream_response(answer: Generator[Any, None, None]) -> None:
    """
    LLM의 스트리밍 응답을 출력하는 함수
    
    Args:
        answer (Generator): LLM의 스트리밍 응답 제너레이터
    """
    try:
        for chunk in answer:
            if hasattr(chunk, 'content'):
                print(chunk.content, end="", flush=True)
            else:
                print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
    finally:
        print()  # 마지막에 줄바꿈 추가

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

examples = [
    {
        "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인
""",
    },
    {
        "question": "네이버의 창립자는 언제 태어났나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 네이버의 창립자는 누구인가요?
중간 답변: 네이버는 이해진에 의해 창립되었습니다.
추가 질문: 이해진은 언제 태어났나요?
중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
최종 답변은: 1967년 6월 22일
""",
    },
    {
        "question": "율곡 이이의 어머니가 태어난 해의 통치하던 왕은 누구인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 율곡 이이의 어머니는 누구인가요?
중간 답변: 율곡 이이의 어머니는 신사임당입니다.
추가 질문: 신사임당은 언제 태어났나요?
중간 답변: 신사임당은 1504년에 태어났습니다.
추가 질문: 1504년에 조선을 통치한 왕은 누구인가요?
중간 답변: 1504년에 조선을 통치한 왕은 연산군입니다.
최종 답변은: 연산군
""",
    },
    {
        "question": "올드보이와 기생충의 감독이 같은 나라 출신인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 올드보이의 감독은 누구인가요?
중간 답변: 올드보이의 감독은 박찬욱입니다.
추가 질문: 박찬욱은 어느 나라 출신인가요?
중간 답변: 박찬욱은 대한민국 출신입니다.
추가 질문: 기생충의 감독은 누구인가요?
중간 답변: 기생충의 감독은 봉준호입니다.
추가 질문: 봉준호는 어느 나라 출신인가요?
중간 답변: 봉준호는 대한민국 출신입니다.
최종 답변은: 예
""",
    },
]

# 객체 생성
llm = ChatOpenAI(
    temperature=0,  # 창의성
    model_name="gpt-4o-mini",  # 모델명
)
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate.from_template(
    "Question:\n{question}\nAnswer:\n{answer}"
)

# 프롬프트 템플릿 생성

prompt = FewShotPromptTemplate(
  examples=examples,
  example_prompt=example_prompt,
  suffix="질문: {question}\n답변:",
  input_variables=["question"],
)

# question = input("질문을 입력해주세요: ")

# final_prompt = prompt.format(question=question)
# answer = llm.stream(final_prompt)

# stream_response(answer)

# ------------------------------------------------------------

# Example Selector
# 예제가 많은 경우, 프롬프트에 포함할 예제를 선택해야 할 수 있다. Example Selector는 이 작업을 담당하는 클래스이다.

from langchain_core.example_selectors import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Vector DB 생성 (저장소 이름, 임베딩 클래스)
chroma = Chroma("example_selector", OpenAIEmbeddings())

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 선택 가능한 예제
    examples,
    # 임베딩 클래스 (의미적 유사성을 측정하는데 사용)
    OpenAIEmbeddings(),
    # 벡터 저장소 클래스 (임베딩을 저장하고, 유사성 검색을 수행)
    Chroma,
    # 선택할 예제 수
    k=1,
)

# 입력과 가장 유사한 예시 선택
question = input("질문을 입력해주세요: ")
selected_examples = example_selector.select_examples({"question": question})

final_prompt = prompt.format(question=question)
answer = llm.stream(final_prompt)

stream_response(answer)
