"""
apps/inference-service/src/inference_service/main.py
FastAPI 서버 - 분류 모델 + 생성 모델 파이프라인 제공
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_service.api.routes import router
from inference_service.classifier.model import ClassifierModel
from inference_service.generator.model import GeneratorModel

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== 분류 모델 로딩 중... ===")
    try:
        models["classifier"] = ClassifierModel.load(
            model_path="../../checkpoints/koelectra_task_cls_result"
        )
        print("=== 분류 모델 로딩 완료 ===")
    except Exception as e:
        print(f"[WARN] 분류 모델 로딩 실패: {e}")

    # 생성 모델 (선택적 — 무거우므로 주석 해제하여 활성화)
    try:
        models["generator"] = GeneratorModel.load(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path="../../checkpoints/qwen_result"
        )
        print("=== 생성 모델 로딩 완료 ===")
    except Exception as e:
        print(f"[WARN] 생성 모델 로딩 실패: {e}")

    app.state.models = models
    yield
    models.clear()


app = FastAPI(
    title="Instruct Classifier API",
    description="민원 상담 분류 + 생성 파이프라인 API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference_service.main:app", host="0.0.0.0", port=8000, reload=True)
