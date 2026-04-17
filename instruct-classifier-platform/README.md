# instruct-classifier-platform

민원 상담 **명시적 인스트럭션 이해** 분류 + 생성 파이프라인 프로젝트

---

## 프로젝트 개요

| 구분 | 내용 |
|---|---|
| 원래 주제 | 인스트럭션 이해 (instruction following) |
| 최종 주제 | **명시적 인스트럭션 이해** — instruction 텍스트로 태스크 라우팅 |
| 분류 모델 | KoELECTRA-base-v3 (분류/요약/질의응답 3-class) |
| 생성 모델 | Qwen2.5-1.5B-Instruct + LoRA |

### 주요 발견 사항

`instruction_only` 모드에서 `full` 모드와 동등하거나 더 높은 성능이 나온 것은 **모델이 input보다 명시적 instruction에 의존한다**는 것을 의미합니다. 이는 사실상 라우터처럼 동작한다는 것이며, 명시적 인스트럭션 이해 관점에서는 긍정적인 결과로 해석할 수 있습니다.

---

## 프로젝트 구조

```
instruct-classifier-platform/
├── apps/
│   ├── inference-service/          ← FastAPI 백엔드
│   │   ├── pyproject.toml
│   │   └── src/inference_service/
│   │       ├── main.py             ← 서버 진입점
│   │       ├── api/
│   │       │   └── routes.py       ← /chat, /classify 엔드포인트
│   │       ├── classifier/
│   │       │   └── model.py        ← KoELECTRA 분류 모델
│   │       └── generator/
│   │           └── model.py        ← Qwen 생성 모델
│   └── frontend-react/             ← React 챗봇 UI
│       ├── index.html
│       ├── package.json
│       ├── vite.config.js
│       └── src/
│           ├── main.jsx
│           └── App.jsx
├── checkpoints/                    ← 학습된 모델 (Colab에서 다운로드)
│   ├── koelectra_task_cls_result/  ← 태스크 분류 모델 (메인)
│   └── qwen_result/                ← Qwen LoRA 어댑터
└── README.md
```

---

## 빠른 시작

### 1. 모델 준비

Colab에서 학습한 모델 폴더를 `checkpoints/`에 복사:

```
checkpoints/
  koelectra_task_cls_result/    ← 필수 (분류)
  qwen_result/                  ← 선택 (생성, 무거움)
```

### 2. 백엔드 실행

```bash
cd apps/inference-service
pip install -e .
cd src
uvicorn inference_service.main:app --reload --port 8000
```

생성 모델 활성화하려면 `main.py`의 주석을 해제:
```python
# models["generator"] = GeneratorModel.load(...)
```

### 3. 프론트엔드 실행

```bash
cd apps/frontend-react
npm install
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

---

## API

### `POST /chat`

챗봇 메인 엔드포인트

```json
{
  "message": "이 민원의 상담 결과를 분류해줘.",
  "instruction": ""
}
```

응답:
```json
{
  "task": "분류",
  "confidence": 0.9823,
  "all_scores": { "분류": 0.9823, "요약": 0.0102, "질의응답": 0.0075 },
  "generated": "입력하신 내용을 분류 태스크로 인식했습니다..."
}
```

### `POST /classify`

순수 분류 엔드포인트 (디버깅용)

---

## 실험 결과 요약

| 모드 | 설명 | val F1 (macro) |
|---|---|---|
| full | instruction + input 모두 사용 | ~0.9998 |
| instruction_only | instruction만 사용 | ~0.9998 |
| input_only | input만 사용 | 성능 저하 |

→ instruction만으로 태스크 분류가 충분히 가능함 = **명시적 instruction이 핵심 신호**
