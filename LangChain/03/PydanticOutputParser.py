# 언어 모델의 출력을 더 구조화된 정보로 변환하는데 도움이 되는 클래스

# 두가지 핵심 메서드가 구현되어야 활용 가능

"""
get_format_instructions(): 언어 모델이 출력해야 할 정보의 형식을 정의하는 지침(instruction) 을 제공합니다. 예를 들어, 언어 모델이 출력해야 할 데이터의 필드와 그 형태를 설명하는 지침을 문자열로 반환할 수 있습니다. 이때 설정하는 지침(instruction) 의 역할이 매우 중요합니다. 이 지침에 따라 언어 모델은 출력을 구조화하고, 이를 특정 데이터 모델에 맞게 변환할 수 있습니다.
parse(): 언어 모델의 출력(문자열로 가정)을 받아들여 이를 특정 구조로 분석하고 변환합니다. Pydantic와 같은 도구를 사용하여, 입력된 문자열을 사전 정의된 스키마에 따라 검증하고, 해당 스키마를 따르는 데이터 구조로 변환합니다.
"""

from dotenv import load_dotenv

load_dotenv()

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


from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

email_conversation = """From: 김철수 (chulsoo.kim@bikecorporation.me)
To: 이은채 (eunchae@teddyinternational.me)
Subject: "ZENESIS" 자전거 유통 협력 및 미팅 일정 제안

안녕하세요, 이은채 대리님,

저는 바이크코퍼레이션의 김철수 상무입니다. 최근 보도자료를 통해 귀사의 신규 자전거 "ZENESIS"에 대해 알게 되었습니다. 바이크코퍼레이션은 자전거 제조 및 유통 분야에서 혁신과 품질을 선도하는 기업으로, 이 분야에서의 장기적인 경험과 전문성을 가지고 있습니다.

ZENESIS 모델에 대한 상세한 브로슈어를 요청드립니다. 특히 기술 사양, 배터리 성능, 그리고 디자인 측면에 대한 정보가 필요합니다. 이를 통해 저희가 제안할 유통 전략과 마케팅 계획을 보다 구체화할 수 있을 것입니다.

또한, 협력 가능성을 더 깊이 논의하기 위해 다음 주 화요일(1월 15일) 오전 10시에 미팅을 제안합니다. 귀사 사무실에서 만나 이야기를 나눌 수 있을까요?

감사합니다.

김철수
상무이사
바이크코퍼레이션
"""

class EmailSummary(BaseModel):
    person: str = Field(description="메일을 보낸 사람")
    email: str = Field(description="메일을 보낸 사람의 이메일 주소")
    subject: str = Field(description="메일의 제목")
    summary: str = Field(description="메일의 본문을 요약한 텍스트")
    data: str = Field(description="메일 본문에 언급된 미팅 날짜와 시간")

# PydanticOutputParser 인스턴스 생성
parser = PydanticOutputParser(pydantic_object=EmailSummary)


# 프롬프트 정의
"""
1. question: 유저의 질문을 받는다.
2. email_conversation: 이메일 본문의 내용을 입력한다.
3. format: 형식을 지정한다.
"""

prompt = PromptTemplate.from_template(
    """
You are a helpful assistant. Plz answer the following questions in Korean.

Question: 
{question}

Email Conversation:
{email_conversation}

FORMAT:
{format}
"""
)

# format 에 PydanticOutputParser의 부분 포맷팅(partial) 추가
prompt = prompt.partial(format=parser.get_format_instructions())

# chain 생성
chain = prompt | llm

# 스트리밍 응답을 문자열로 수집
response_text = ""
response = chain.stream({
    "email_conversation": email_conversation,
    "question": "이메일 내용 중 주요 내용을 추출해 주세요."
})

# 스트리밍 응답 출력 및 수집
for chunk in response:
    if hasattr(chunk, 'content'):
        print(chunk.content, end="", flush=True)
        response_text += chunk.content
    else:
        print(chunk, end="", flush=True)
        response_text += chunk
print()  # 줄바꿈 추가

# 수집된 응답을 파싱
result = parser.parse(response_text)
print("\n파싱된 결과:")
print(result)