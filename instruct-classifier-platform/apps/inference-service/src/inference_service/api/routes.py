"""
apps/inference-service/src/inference_service/api/routes.py
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    message: str           # 사용자 자연어 입력 (instruction 없이도 동작)
    instruction: str = ""  # 선택적 명시 instruction


class ChatResponse(BaseModel):
    task: str              # 분류 / 요약 / 질의응답
    confidence: float
    all_scores: dict[str, float]
    generated: str         # 생성 모델 출력 (없으면 빈 문자열)


class ClassifyRequest(BaseModel):
    instruction: str = ""
    user_input: str = ""


class ClassifyResponse(BaseModel):
    task: str
    confidence: float
    all_scores: dict[str, float]


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    챗봇 엔드포인트.
    사용자 자연어 메시지 → 태스크 분류 → 생성 모델 응답
    """
    classifier = request.app.state.models.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="분류 모델이 로드되지 않았습니다.")

    # instruction이 없으면 message 자체를 input으로 사용
    instruction = req.instruction.strip()
    user_input = req.message.strip()

    cls_result = classifier.predict(
        instruction=instruction,
        user_input=user_input,
    )

    generator = request.app.state.models.get("generator")
    generated = ""
    if generator is not None:
        generated = generator.generate(
            task=cls_result["task"],
            instruction=instruction,
            user_input=user_input,
        )
    else:
        # 생성 모델 미활성화 시 태스크별 안내 메시지
        task = cls_result["task"]
        generated = _fallback_response(task, user_input)

    return ChatResponse(
        task=cls_result["task"],
        confidence=cls_result["confidence"],
        all_scores=cls_result["all_scores"],
        generated=generated,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest, request: Request):
    """순수 분류 엔드포인트 (디버깅/데모용)"""
    classifier = request.app.state.models.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="분류 모델이 로드되지 않았습니다.")

    result = classifier.predict(
        instruction=req.instruction,
        user_input=req.user_input,
    )
    return ClassifyResponse(**result)


def _fallback_response(task: str, user_input: str) -> str:
    """생성 모델 미로드 시 태스크별 안내 메시지"""
    if task == "분류":
        return (
            "입력하신 내용을 분류 태스크로 인식했습니다. "
            "생성 모델이 활성화되면 상담 결과, 주제, 내용 등을 자동으로 분류하여 답변드립니다."
        )
    elif task == "요약":
        return (
            "입력하신 내용을 요약 태스크로 인식했습니다. "
            "생성 모델이 활성화되면 핵심 내용을 요약하여 제공합니다."
        )
    elif task == "질의응답":
        return (
            "입력하신 내용을 질의응답 태스크로 인식했습니다. "
            "생성 모델이 활성화되면 질문에 대한 답변을 생성합니다."
        )
    else:
        return "생성 모델이 활성화되어 있지 않습니다. 백엔드 설정을 확인해주세요."