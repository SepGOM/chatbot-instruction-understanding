# Chatbot Instruction Understanding

한국어 공공기관 상담 데이터를 활용한 instruction-following 챗봇 전처리 파이프라인

---

## 프로젝트 구조

```
chatbot-instruction-understanding/
├── src/
│   └── preprocess_pipeline.py     # 전처리 파이프라인
├── notebooks/
│   └── Project2_data_convert.ipynb  # 데이터 포맷 변환
├── data/
│   ├── processed/                 # 전처리 완료 데이터 (gitignore, 로컬 생성)
│   └── sample/                    # 포맷 예시 샘플
└── Daily Standup/                 # 팀 미팅 기록
```

---

## 파이프라인 설명

### 1. `src/preprocess_pipeline.py`

AI-Hub 공공기관 민원 라벨링 데이터를 LLM 학습에 적합한 형태로 전처리하는 파이프라인.

**주요 기능:**
- **텍스트 정제**: 유니코드 NFC 정규화, HTML 태그 제거, 제어문자 제거, 인사말 제거
- **PII 마스킹**: 전화번호, 이메일, 주민등록번호, 계좌번호, 주소 탐지 및 마스킹
- **품질 필터링**: 길이 제약, 반복 비율, 빈 필드 처리
- **중복 제거**: SHA256 해시(완전 중복) + MinHash+LSH(유사 중복, Jaccard 0.85)
- **출력 포맷**: Alpaca 형식 또는 JSONL 형식 선택 가능

### 2. `notebooks/Project2_data_convert.ipynb`

전처리된 Alpaca 3-field 데이터를 OpenAI chat 형식(messages 배열)으로 변환.

**변환 포맷:**
```json
{
  "messages": [
    {"role": "system", "content": "당신은 사용자의 지시를 이해하고 적절한 형식으로 한국어 응답을 생성하는 도우미이다."},
    {"role": "user",   "content": "[instruction]\n[input]"},
    {"role": "assistant", "content": "[output]"}
  ]
}
```

---

## 데이터

### 원본 데이터 출처

[AI-Hub - 공공기관 민원 안내 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71852)

> **라벨링 데이터만 사용** (원천 데이터 미사용)  
> AI-Hub 이용약관에 따라 원본 데이터를 이 레포에 포함하지 않음.  
> 재현을 위해서는 AI-Hub 가입 후 아래 데이터셋을 직접 다운로드하세요.

사용 데이터셋:
- `TL_국립아시아문화전당_분류`, `TL_국립아시아문화전당_요약`, `TL_국립아시아문화전당_질의응답`
- `TL_중앙행정기관_분류`, `TL_중앙행정기관_요약`, `TL_중앙행정기관_질의응답`
- `TL_지방행정기관_분류`, `TL_지방행정기관_요약`, `TL_지방행정기관_질의응답`

### Processed 데이터

전처리 완료 데이터는 용량 문제로 레포에 포함하지 않습니다 (train: ~291MB, val: ~39MB).  
아래 실행 방법으로 로컬에서 생성하세요.

---

## 실행 방법

```bash
# 1. AI-Hub에서 라벨링 데이터 다운로드 후 raw_data/ 폴더에 위치
#    raw_data/Training/TL_*/

# 2. 전처리 파이프라인 실행
python src/preprocess_pipeline.py

# 3. 데이터 포맷 변환 (Alpaca → chat 형식)
#    notebooks/Project2_data_convert.ipynb 순서대로 실행
```

---

## 환경

```bash
pip install datasketch  # MinHash+LSH 중복 제거
```
