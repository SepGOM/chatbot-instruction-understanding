"""
=============================================================================
공공 민원 상담 LLM 사전학습 데이터 — 전처리 파이프라인
=============================================================================
용도: 사전학습 전용 (Alpaca 포맷 출력, 추후 변환 가능)
기능: EDA → 텍스트 정제 → PII 탐지 → 품질 필터링 → 중복 제거 → 출력
각 단계별 강도(intensity) 파라미터를 config에서 세부 조절 가능
=============================================================================
"""

import json
import re
import hashlib
import os
import glob
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime


# ============================================================================
# 0. CONFIG — 각 전처리 단계별 강도(intensity) 세부 설정
# ============================================================================

@dataclass
class PipelineConfig:
    """전처리 파이프라인 전체 설정. 각 단계별 강도를 세부 조절할 수 있습니다."""

    # --- 입출력 경로 ---
    train_dir: str = "./raw_data/Training"      # AI허브 Training 폴더 경로
    val_dir: str = "./raw_data/Validation"      # AI허브 Validation 폴더 경로
    output_dir: str = "./processed_data"        # 전처리 결과 저장 디렉토리
    report_dir: str = "./reports"               # EDA 리포트 저장 디렉토리

    # --- [1단계] 텍스트 정제 강도 ---
    clean_normalize_whitespace: bool = True      # 연속 공백/개행 정규화
    clean_strip_html_tags: bool = True           # HTML 태그 제거
    clean_normalize_unicode: bool = True         # 유니코드 NFC 정규화
    clean_remove_control_chars: bool = True      # 제어 문자 제거 (\x00-\x1f 등)
    clean_collapse_newlines: int = 2             # 최대 연속 개행 수 (0=미적용)
    clean_strip_leading_numbering: bool = False  # "1. ", "2. " 등 문두 번호 제거 (보수적: off)
    clean_remove_greeting_phrases: bool = True   # 정형 인사말/안내 문구 제거
    greeting_patterns: List[str] = field(default_factory=lambda: [
        r"안녕하세요[.\s]*",
        r"업무에 고생이 많으십니다[.\s]*",
        r"평소 시정에 관심을 갖고 건의하여 주심에 감사드립니다[.\s]*",
        r"감사합니다[.\s]*$",
    ])

    # --- [2단계] PII 탐지 강도 ---
    pii_check_mask_completeness: bool = True     # ▲ 마스킹 패턴 검증
    pii_detect_phone: bool = True                # 전화번호 탐지
    pii_detect_email: bool = True                # 이메일 탐지
    pii_detect_rrn: bool = True                  # 주민등록번호 탐지
    pii_detect_account: bool = True              # 계좌번호 탐지
    pii_detect_address: bool = True              # 상세주소 패턴 탐지
    pii_action: str = "flag"                     # "flag" (리포트만) | "mask" (자동 마스킹) | "drop" (해당 샘플 제거)
    pii_mask_token: str = "[PII]"                # 자동 마스킹 시 대체 토큰

    # --- [3단계] 품질 필터링 강도 ---
    filter_min_instruction_len: int = 5          # instruction 최소 글자 수
    filter_min_input_len: int = 20               # input 최소 글자 수
    filter_min_output_len: int = 5               # output 최소 글자 수 (요약 등 기본)
    filter_max_input_len: int = 10000            # input 최대 글자 수
    filter_max_output_len: int = 5000            # output 최대 글자 수
    filter_drop_empty_fields: bool = True        # 빈 필드 샘플 제거
    filter_min_output_word_count: int = 2        # output 최소 어절 수 (요약 등 기본)
    filter_max_repetition_ratio: float = 0.5     # 동일 문장 반복 비율 상한 (0.0~1.0)
    filter_drop_if_input_equals_output: bool = True  # input == output인 경우 제거

    # --- [3-1단계] 단답형 output 허용 태스크 전용 필터 ---
    qa_filter_min_output_len: int = 1            # 단답 허용 태스크 output 최소 글자 수
    qa_filter_min_output_word_count: int = 1     # 단답 허용 태스크 output 최소 어절 수
    qa_task_names: List[str] = field(default_factory=lambda: ["질의응답", "분류"])  # 단답형 output을 허용할 태스크 목록

    # --- [4단계] 중복 제거 강도 ---
    dedup_exact: bool = True                     # 정확 중복 제거 (SHA256)
    dedup_near: bool = True                      # 유사 중복 제거
    dedup_near_threshold: float = 0.85           # 유사도 임계값 (0.0~1.0, 높을수록 보수적)
    dedup_ngram_size: int = 5                    # near-dup용 n-gram 크기
    dedup_num_perm: int = 128                    # MinHash 퍼뮤테이션 수 (정확도↑ = 느림)
    dedup_field: str = "instruction_output"     # 중복 판단 기준 ("input" | "output" | "both" | "instruction_output")
    dedup_near_skip_tasks: List[str] = field(default_factory=lambda: ["분류"])  # near-dup 제외 태스크 (exact dedup만 적용)
    dedup_skip_task_field: str = "input_output"  # near-dup 제외 태스크의 exact dedup 기준 ("input_output" = input+output 결합)

    # --- [5단계] 출력 설정 ---
    output_format: str = "alpaca"                # "alpaca" | "jsonl"
    include_metadata: bool = True                # task, category 등 메타 포함 여부

    # --- 로깅 ---
    log_level: str = "INFO"
    log_file: Optional[str] = "pipeline.log"


# ============================================================================
# 1. 로거 설정
# ============================================================================

def setup_logger(config: PipelineConfig) -> logging.Logger:
    logger = logging.getLogger("preprocess")
    logger.setLevel(getattr(logging, config.log_level))
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if config.log_file:
        os.makedirs(os.path.dirname(config.log_file) if os.path.dirname(config.log_file) else ".", exist_ok=True)
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# 2. 데이터 로딩 — 중첩 JSON → flat records
# ============================================================================

def load_and_flatten(input_dir: str, config: PipelineConfig, logger: logging.Logger) -> List[Dict[str, Any]]:
    """원본 JSON 파일들을 읽어 flat한 레코드 리스트로 변환합니다."""
    records = []
    json_files = sorted(glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True))

    if not json_files:
        logger.warning(f"No JSON files found in {input_dir}")
        return records

    logger.info(f"Found {len(json_files)} JSON file(s) in {input_dir}")

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"Failed to load {fpath}: {e}")
            continue

        # 파일이 리스트가 아닌 경우 리스트로 감싸기
        if isinstance(data, dict):
            data = [data]

        for doc in data:
            # 메타데이터 추출
            meta = {
                "source": doc.get("source", ""),
                "consulting_category": doc.get("consulting_category", ""),
                "consulting_content": doc.get("consulting_content", ""),
                "file_origin": os.path.basename(fpath),
            }

            # instructions 배열 순회
            instructions_list = doc.get("instructions", [])
            if not isinstance(instructions_list, list):
                instructions_list = [instructions_list]

            for inst_group in instructions_list:
                tuning_type = inst_group.get("tuning_type", "")
                data_items = inst_group.get("data", [])

                if not isinstance(data_items, list):
                    data_items = [data_items]

                for item in data_items:
                    record = {
                        "instruction": item.get("instruction", ""),
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                        "task": item.get("task", tuning_type),
                        "task_category": item.get("task_category", ""),
                        **meta,
                    }
                    records.append(record)

    logger.info(f"Loaded {len(records)} instruction records from {len(json_files)} files")
    return records


# ============================================================================
# 3. EDA — 통계 리포트 생성
# ============================================================================

def run_eda(records: List[Dict], config: PipelineConfig, logger: logging.Logger, stage: str = "raw") -> Dict:
    """데이터셋 통계를 수집하고 리포트를 생성합니다."""
    logger.info(f"[EDA-{stage}] Running statistics on {len(records)} records...")

    stats = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "total_records": len(records),
        "field_stats": {},
        "task_distribution": dict(Counter(r.get("task", "") for r in records)),
        "task_category_distribution": dict(Counter(r.get("task_category", "") for r in records)),
        "source_distribution": dict(Counter(r.get("source", "") for r in records)),
        "category_distribution": dict(Counter(r.get("consulting_category", "") for r in records)),
    }

    # 필드별 길이 통계
    for field_name in ["instruction", "input", "output"]:
        lengths = [len(r.get(field_name, "")) for r in records]
        word_counts = [len(r.get(field_name, "").split()) for r in records]
        empty_count = sum(1 for l in lengths if l == 0)

        if lengths:
            stats["field_stats"][field_name] = {
                "count": len(lengths),
                "empty_count": empty_count,
                "empty_ratio": round(empty_count / len(lengths), 4),
                "char_len": {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": round(sum(lengths) / len(lengths), 1),
                    "median": sorted(lengths)[len(lengths) // 2],
                },
                "word_count": {
                    "min": min(word_counts),
                    "max": max(word_counts),
                    "mean": round(sum(word_counts) / len(word_counts), 1),
                    "median": sorted(word_counts)[len(word_counts) // 2],
                },
            }

    # 리포트 저장
    os.makedirs(config.report_dir, exist_ok=True)
    report_path = os.path.join(config.report_dir, f"eda_{stage}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"[EDA-{stage}] Report saved to {report_path}")
    _print_eda_summary(stats, logger)

    return stats


def _print_eda_summary(stats: Dict, logger: logging.Logger):
    """EDA 결과를 콘솔에 요약 출력합니다."""
    logger.info(f"  ├─ Total records: {stats['total_records']}")
    logger.info(f"  ├─ Task distribution: {stats['task_distribution']}")
    logger.info(f"  ├─ Source distribution: {stats['source_distribution']}")

    for fname, fstat in stats.get("field_stats", {}).items():
        cl = fstat["char_len"]
        logger.info(
            f"  ├─ [{fname}] empty={fstat['empty_count']}({fstat['empty_ratio']:.1%}), "
            f"chars: min={cl['min']} / med={cl['median']} / max={cl['max']} / mean={cl['mean']}"
        )


# ============================================================================
# 4. 텍스트 정제 (Text Cleaning)
# ============================================================================

def clean_text(text: str, config: PipelineConfig) -> str:
    """텍스트 정제를 설정에 따라 수행합니다."""
    if not text:
        return text

    # 유니코드 NFC 정규화
    if config.clean_normalize_unicode:
        import unicodedata
        text = unicodedata.normalize("NFC", text)

    # 제어 문자 제거
    if config.clean_remove_control_chars:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # HTML 태그 제거
    if config.clean_strip_html_tags:
        text = re.sub(r"<[^>]+>", " ", text)

    # 정형 인사말/안내 문구 제거
    if config.clean_remove_greeting_phrases:
        for pattern in config.greeting_patterns:
            text = re.sub(pattern, "", text)

    # 연속 공백 정규화
    if config.clean_normalize_whitespace:
        text = re.sub(r"[^\S\n]+", " ", text)  # 개행 외 연속 공백 → 단일 공백
        text = re.sub(r" *\n *", "\n", text)    # 개행 전후 공백 제거

    # 연속 개행 제한
    if config.clean_collapse_newlines > 0:
        max_nl = config.clean_collapse_newlines
        text = re.sub(r"\n{" + str(max_nl + 1) + r",}", "\n" * max_nl, text)

    # 문두 번호 제거
    if config.clean_strip_leading_numbering:
        text = re.sub(r"^(\d+)\.\s+", "", text, flags=re.MULTILINE)

    return text.strip()


def apply_cleaning(records: List[Dict], config: PipelineConfig, logger: logging.Logger) -> List[Dict]:
    """모든 레코드의 텍스트 필드에 정제를 적용합니다."""
    logger.info("[CLEAN] Applying text cleaning...")
    text_fields = ["instruction", "input", "output", "consulting_content"]

    for record in records:
        for field_name in text_fields:
            if field_name in record and record[field_name]:
                record[field_name] = clean_text(record[field_name], config)

    logger.info(f"[CLEAN] Cleaned {len(records)} records across {len(text_fields)} fields")
    return records


# ============================================================================
# 5. PII 탐지 및 처리
# ============================================================================

# PII 패턴 정의
PII_PATTERNS = {
    "phone": re.compile(
        r"(?<!\d)"                         # 앞에 숫자가 아닌 것
        r"(0\d{1,2})"                      # 지역번호/휴대폰 앞자리
        r"[-.\s]?"
        r"(\d{3,4})"                       # 중간 번호
        r"[-.\s]?"
        r"(\d{4})"                         # 끝 번호
        r"(?!\d)"                          # 뒤에 숫자가 아닌 것
    ),
    "email": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "rrn": re.compile(
        r"\d{6}[-\s]?[1-4]\d{6}"          # 주민등록번호
    ),
    "account": re.compile(
        r"\d{3,6}[-\s]?\d{2,6}[-\s]?\d{2,6}[-\s]?\d{0,6}"  # 계좌번호 패턴
    ),
    "address_detail": re.compile(
        r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|전북|전남|경북|경남|제주)"
        r"(?:특별시|광역시|특별자치시|도|특별자치도)?\s*"
        r"(?:\S+[시군구])\s*"
        r"(?:\S+[동읍면리로길])\s*"
        r"\d+"
    ),
}


def detect_pii(text: str, config: PipelineConfig) -> List[Dict[str, str]]:
    """텍스트에서 PII를 탐지합니다. 반환: [{"type": ..., "match": ..., "span": ...}, ...]"""
    findings = []

    if not text:
        return findings

    # ▲ 마스킹 완전성 검증
    if config.pii_check_mask_completeness:
        # 마스킹 사이에 남아있는 실제 정보 패턴 탐지
        # 예: "▲▲▲ 홍길동 ▲▲▲" 에서 "홍길동"이 노출된 경우
        mask_gaps = re.finditer(r"▲+\s+([가-힣]{2,4})\s+▲+", text)
        for m in mask_gaps:
            findings.append({
                "type": "mask_leak_name",
                "match": m.group(1),
                "span": f"{m.start()}-{m.end()}",
            })

    pattern_map = {
        "phone": config.pii_detect_phone,
        "email": config.pii_detect_email,
        "rrn": config.pii_detect_rrn,
        "account": config.pii_detect_account,
        "address_detail": config.pii_detect_address,
    }

    for pii_type, enabled in pattern_map.items():
        if not enabled:
            continue
        for m in PII_PATTERNS[pii_type].finditer(text):
            findings.append({
                "type": pii_type,
                "match": m.group(),
                "span": f"{m.start()}-{m.end()}",
            })

    return findings


def apply_pii_processing(
    records: List[Dict], config: PipelineConfig, logger: logging.Logger
) -> Tuple[List[Dict], List[Dict]]:
    """PII 탐지 후 설정된 액션(flag/mask/drop)을 수행합니다."""
    logger.info(f"[PII] Scanning PII (action={config.pii_action})...")

    pii_report = []
    clean_records = []
    flagged_count = 0
    dropped_count = 0

    text_fields = ["instruction", "input", "output"]

    for idx, record in enumerate(records):
        record_findings = []

        for field_name in text_fields:
            text = record.get(field_name, "")
            findings = detect_pii(text, config)
            for f in findings:
                f["record_idx"] = idx
                f["field"] = field_name
            record_findings.extend(findings)

        if not record_findings:
            clean_records.append(record)
            continue

        flagged_count += 1
        pii_report.extend(record_findings)

        if config.pii_action == "drop":
            dropped_count += 1
            continue

        elif config.pii_action == "mask":
            for field_name in text_fields:
                text = record.get(field_name, "")
                for finding in record_findings:
                    if finding["field"] == field_name:
                        text = text.replace(finding["match"], config.pii_mask_token)
                record[field_name] = text
            clean_records.append(record)

        else:  # "flag"
            record["_pii_flags"] = record_findings
            clean_records.append(record)

    # PII 리포트 저장
    os.makedirs(config.report_dir, exist_ok=True)
    pii_report_path = os.path.join(config.report_dir, "pii_report.json")
    with open(pii_report_path, "w", encoding="utf-8") as f:
        json.dump(pii_report, f, ensure_ascii=False, indent=2)

    logger.info(f"[PII] Scanned {len(records)} records → flagged={flagged_count}, dropped={dropped_count}")
    logger.info(f"[PII] PII detail report saved to {pii_report_path}")

    # PII 유형별 통계
    type_counts = Counter(f["type"] for f in pii_report)
    for pii_type, count in type_counts.most_common():
        logger.info(f"  ├─ {pii_type}: {count} occurrences")

    return clean_records, pii_report


# ============================================================================
# 6. 품질 필터링 (Quality Filtering)
# ============================================================================

def compute_repetition_ratio(text: str) -> float:
    """문장 단위 반복 비율을 계산합니다."""
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    if len(sentences) <= 1:
        return 0.0
    counter = Counter(sentences)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(sentences)


def apply_quality_filter(
    records: List[Dict], config: PipelineConfig, logger: logging.Logger
) -> Tuple[List[Dict], Dict[str, int]]:
    """품질 기준에 따라 샘플을 필터링합니다. 질의응답 태스크는 단답형을 허용하는 완화된 기준을 적용합니다."""
    logger.info("[FILTER] Applying quality filters...")
    logger.info(f"  ├─ QA relaxed mode: min_output_len={config.qa_filter_min_output_len}, "
                f"min_output_words={config.qa_filter_min_output_word_count}")

    drop_reasons = Counter()
    passed = []

    for record in records:
        instruction = record.get("instruction", "")
        inp = record.get("input", "")
        out = record.get("output", "")
        task = record.get("task", "")

        # 태스크별 기준 분기
        is_qa = task in config.qa_task_names
        min_output_len = config.qa_filter_min_output_len if is_qa else config.filter_min_output_len
        min_output_words = config.qa_filter_min_output_word_count if is_qa else config.filter_min_output_word_count

        # 빈 필드 체크
        if config.filter_drop_empty_fields:
            if not instruction or not inp or not out:
                drop_reasons["empty_field"] += 1
                continue

        # 최소 길이
        if len(instruction) < config.filter_min_instruction_len:
            drop_reasons["instruction_too_short"] += 1
            continue
        if len(inp) < config.filter_min_input_len:
            drop_reasons["input_too_short"] += 1
            continue
        if len(out) < min_output_len:
            drop_reasons[f"output_too_short({'qa' if is_qa else 'default'})"] += 1
            continue

        # 최대 길이
        if len(inp) > config.filter_max_input_len:
            drop_reasons["input_too_long"] += 1
            continue
        if len(out) > config.filter_max_output_len:
            drop_reasons["output_too_long"] += 1
            continue

        # output 어절 수
        if len(out.split()) < min_output_words:
            drop_reasons[f"output_too_few_words({'qa' if is_qa else 'default'})"] += 1
            continue

        # input == output 동일
        if config.filter_drop_if_input_equals_output and inp.strip() == out.strip():
            drop_reasons["input_equals_output"] += 1
            continue

        # 반복 비율
        if config.filter_max_repetition_ratio > 0:
            rep_ratio = compute_repetition_ratio(out)
            if rep_ratio > config.filter_max_repetition_ratio:
                drop_reasons["output_too_repetitive"] += 1
                continue

        passed.append(record)

    total_dropped = sum(drop_reasons.values())
    logger.info(f"[FILTER] {len(records)} → {len(passed)} records (dropped {total_dropped})")
    for reason, count in drop_reasons.most_common():
        logger.info(f"  ├─ {reason}: {count}")

    # 필터 리포트 저장
    os.makedirs(config.report_dir, exist_ok=True)
    filter_report_path = os.path.join(config.report_dir, "filter_report.json")
    with open(filter_report_path, "w", encoding="utf-8") as f:
        json.dump({"total_input": len(records), "total_output": len(passed),
                    "drop_reasons": dict(drop_reasons)}, f, ensure_ascii=False, indent=2)

    return passed, dict(drop_reasons)


# ============================================================================
# 7. 중복 제거 (Deduplication)
# ============================================================================

def get_dedup_text(record: Dict, config: PipelineConfig, field_override: str = None) -> str:
    """중복 판단 기준 텍스트를 추출합니다. field_override가 주어지면 config 대신 사용."""
    field = field_override or config.dedup_field
    if field == "input":
        return record.get("input", "")
    elif field == "output":
        return record.get("output", "")
    elif field == "both":
        return record.get("input", "") + " ||| " + record.get("output", "")
    elif field == "instruction_output":
        return record.get("instruction", "") + " ||| " + record.get("output", "")
    elif field == "input_output":
        return record.get("input", "") + " ||| " + record.get("output", "")
    return record.get("input", "")


def exact_dedup(records: List[Dict], config: PipelineConfig, logger: logging.Logger,
                field_override: str = None) -> List[Dict]:
    """SHA256 해시 기반 정확 중복 제거. field_override로 기준 필드를 변경할 수 있음."""
    field_used = field_override or config.dedup_field
    logger.info(f"[DEDUP-EXACT] Running exact deduplication (field={field_used})...")
    seen_hashes: Set[str] = set()
    unique = []

    for record in records:
        text = get_dedup_text(record, config, field_override=field_override)
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(record)

    removed = len(records) - len(unique)
    logger.info(f"[DEDUP-EXACT] {len(records)} → {len(unique)} (removed {removed} exact duplicates)")
    return unique


def get_ngrams(text: str, n: int) -> Set[str]:
    """텍스트에서 n-gram 셋을 추출합니다."""
    tokens = text.split()
    if len(tokens) < n:
        return {text}
    return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """두 집합의 Jaccard 유사도를 계산합니다."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def near_dedup(records: List[Dict], config: PipelineConfig, logger: logging.Logger) -> List[Dict]:
    """
    MinHash + LSH 기반 유사 중복 제거 (순수 Python 구현).
    O(n) 수준으로 동작하여 50,000건 이상에서도 수 분 내 완료됩니다.

    작동 원리:
    1. 각 레코드의 텍스트에서 n-gram을 추출
    2. MinHash 서명(num_perm개의 해시값)을 생성
    3. LSH(Locality Sensitive Hashing)로 후보 쌍을 빠르게 찾음
    4. 후보 쌍에 대해서만 실제 Jaccard 유사도를 계산
    """
    import struct
    import random as _random

    num_perm = config.dedup_num_perm
    threshold = config.dedup_near_threshold
    ngram_size = config.dedup_ngram_size

    logger.info(f"[DEDUP-NEAR] Running MinHash+LSH near-dedup "
                f"(threshold={threshold}, ngram={ngram_size}, num_perm={num_perm})...")

    # --- Step 1: 해시 함수 계수 생성 (a*x + b mod p) ---
    _MERSENNE_PRIME = (1 << 61) - 1
    _MAX_HASH = (1 << 32) - 1
    rng = _random.Random(42)
    hash_a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    hash_b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]

    def _hash_token(token: str) -> int:
        """토큰을 32비트 정수로 해싱합니다."""
        return struct.unpack('<I', hashlib.md5(token.encode('utf-8')).digest()[:4])[0]

    def _compute_minhash(ngrams: Set[str]) -> List[int]:
        """n-gram 집합에서 MinHash 서명을 생성합니다."""
        if not ngrams:
            return [_MAX_HASH] * num_perm

        hashed = [_hash_token(ng) for ng in ngrams]
        signature = []
        for i in range(num_perm):
            min_val = _MAX_HASH
            a, b = hash_a[i], hash_b[i]
            for h in hashed:
                val = ((a * h + b) % _MERSENNE_PRIME) & _MAX_HASH
                if val < min_val:
                    min_val = val
            signature.append(min_val)
        return signature

    def _estimate_jaccard(sig_a: List[int], sig_b: List[int]) -> float:
        """두 MinHash 서명의 추정 Jaccard 유사도를 계산합니다."""
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    # --- Step 2: LSH 밴드 수 계산 ---
    # b bands × r rows = num_perm, threshold ≈ (1/b)^(1/r)
    best_b, best_r = 1, num_perm
    best_diff = float('inf')
    for b in range(1, num_perm + 1):
        if num_perm % b != 0:
            continue
        r = num_perm // b
        approx_t = (1.0 / b) ** (1.0 / r)
        diff = abs(approx_t - threshold)
        if diff < best_diff:
            best_diff = diff
            best_b, best_r = b, r

    num_bands = best_b
    rows_per_band = best_r
    logger.info(f"  ├─ LSH config: {num_bands} bands × {rows_per_band} rows "
                f"(approx threshold={(1.0/num_bands)**(1.0/rows_per_band):.3f})")

    # --- Step 3: n-gram 추출 + MinHash 생성 ---
    logger.info(f"  ├─ Computing MinHash signatures for {len(records)} records...")
    signatures = []
    ngram_sets = []
    for record in records:
        text = get_dedup_text(record, config)
        ngrams = get_ngrams(text, ngram_size)
        ngram_sets.append(ngrams)
        signatures.append(_compute_minhash(ngrams))

    # --- Step 4: LSH 버킷팅 ---
    logger.info(f"  ├─ Building LSH index...")
    buckets: Dict[tuple, List[int]] = defaultdict(list)
    for idx, sig in enumerate(signatures):
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_hash = tuple(sig[start:end])
            bucket_key = (band_idx, band_hash)
            buckets[bucket_key].append(idx)

    # --- Step 5: 후보 쌍 추출 + 실제 Jaccard 검증 ---
    logger.info(f"  ├─ Finding candidate pairs from {len(buckets)} buckets...")
    is_duplicate = [False] * len(records)
    dup_count = 0
    checked_pairs: Set[tuple] = set()

    for bucket_indices in buckets.values():
        if len(bucket_indices) < 2:
            continue
        for i_pos in range(len(bucket_indices)):
            idx_i = bucket_indices[i_pos]
            if is_duplicate[idx_i]:
                continue
            for j_pos in range(i_pos + 1, len(bucket_indices)):
                idx_j = bucket_indices[j_pos]
                if is_duplicate[idx_j]:
                    continue

                pair_key = (idx_i, idx_j)
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # 실제 Jaccard 계산 (MinHash 추정 대신 정확한 값)
                sim = jaccard_similarity(ngram_sets[idx_i], ngram_sets[idx_j])
                if sim >= threshold:
                    is_duplicate[idx_j] = True
                    dup_count += 1

    unique = [r for r, dup in zip(records, is_duplicate) if not dup]
    logger.info(f"  ├─ Candidate pairs checked: {len(checked_pairs)}")
    logger.info(f"[DEDUP-NEAR] {len(records)} → {len(unique)} (removed {dup_count} near-duplicates)")
    return unique


def apply_deduplication(records: List[Dict], config: PipelineConfig, logger: logging.Logger) -> List[Dict]:
    """설정에 따라 중복 제거를 적용합니다.
    near-dup 제외 태스크(분류 등)는 exact dedup만 적용하고,
    나머지 태스크는 exact + near-dup을 모두 적용합니다."""

    if not config.dedup_near_skip_tasks:
        # 제외 태스크가 없으면 기존 로직 그대로
        if config.dedup_exact:
            records = exact_dedup(records, config, logger)
        if config.dedup_near:
            records = near_dedup(records, config, logger)
        return records

    # --- 태스크별 분리 ---
    skip_tasks = set(config.dedup_near_skip_tasks)
    records_skip = [r for r in records if r.get("task", "") in skip_tasks]
    records_normal = [r for r in records if r.get("task", "") not in skip_tasks]

    skip_task_names = ", ".join(skip_tasks)
    logger.info(f"[DEDUP] Splitting by task: near-dup 대상={len(records_normal)}, "
                f"exact-only ({skip_task_names})={len(records_skip)}")

    # exact dedup: 양쪽 모두 적용 (분류는 input_output 기준)
    if config.dedup_exact:
        records_normal = exact_dedup(records_normal, config, logger)
        if records_skip:
            logger.info(f"[DEDUP] Running exact dedup on {skip_task_names} (field={config.dedup_skip_task_field})...")
            records_skip = exact_dedup(records_skip, config, logger,
                                       field_override=config.dedup_skip_task_field)

    # near-dup: 일반 태스크에만 적용
    if config.dedup_near and records_normal:
        records_normal = near_dedup(records_normal, config, logger)

    # 합치기
    combined = records_normal + records_skip
    logger.info(f"[DEDUP] Combined: {len(records_normal)} (normal) + {len(records_skip)} ({skip_task_names}) "
                f"= {len(combined)} total")
    return combined


# ============================================================================
# 8. 출력 — Alpaca 포맷 변환 및 저장
# ============================================================================

def format_alpaca(record: Dict, config: PipelineConfig) -> Dict:
    """레코드를 Alpaca 포맷으로 변환합니다."""
    output = {
        "instruction": record["instruction"],
        "input": record["input"],
        "output": record["output"],
    }

    if config.include_metadata:
        output["task"] = record.get("task", "")
        output["task_category"] = record.get("task_category", "")
        output["source"] = record.get("source", "")
        output["consulting_category"] = record.get("consulting_category", "")

    return output


def save_output(records: List[Dict], config: PipelineConfig, logger: logging.Logger, split_name: str):
    """최종 데이터를 split 이름으로 저장합니다."""

    os.makedirs(config.output_dir, exist_ok=True)

    # Alpaca 포맷 변환
    formatted = [format_alpaca(r, config) for r in records]

    out_path = os.path.join(config.output_dir, f"{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    logger.info(f"[OUTPUT] Saved {split_name}: {len(formatted)} records → {out_path}")


# ============================================================================
# 9. 파이프라인 실행 — 메인 오케스트레이터
# ============================================================================

def process_split(
    split_name: str, input_dir: str, config: PipelineConfig, logger: logging.Logger
) -> Optional[List[Dict]]:
    """단일 split(train 또는 val)에 대해 전처리를 수행합니다."""

    logger.info("")
    logger.info(f"{'─' * 30} [{split_name.upper()}] {'─' * 30}")

    # 데이터 로딩
    records = load_and_flatten(input_dir, config, logger)
    if not records:
        logger.warning(f"[{split_name}] No records loaded from {input_dir}. Skipping.")
        return None

    # EDA (전처리 전)
    run_eda(records, config, logger, stage=f"{split_name}_01_raw")

    # 텍스트 정제
    records = apply_cleaning(records, config, logger)

    # PII 탐지 및 처리
    records, pii_report = apply_pii_processing(records, config, logger)

    # 품질 필터링
    records, filter_report = apply_quality_filter(records, config, logger)

    # 중복 제거
    records = apply_deduplication(records, config, logger)

    # EDA (전처리 후)
    run_eda(records, config, logger, stage=f"{split_name}_02_processed")

    # 저장
    save_output(records, config, logger, split_name=split_name)

    return records


def run_pipeline(config: PipelineConfig = None):
    """전처리 파이프라인을 순차적으로 실행합니다.
    AI허브 제공 Training/Validation split을 그대로 유지하며 각각 독립 전처리합니다."""

    if config is None:
        config = PipelineConfig()

    logger = setup_logger(config)

    logger.info("=" * 70)
    logger.info("공공 민원 상담 LLM 데이터 전처리 파이프라인 시작")
    logger.info("=" * 70)
    logger.info(f"  Train dir : {config.train_dir}")
    logger.info(f"  Val dir   : {config.val_dir}")
    logger.info(f"  Output dir: {config.output_dir}")

    # --- Training split 처리 ---
    train_records = process_split("train", config.train_dir, config, logger)

    # --- Validation split 처리 ---
    val_records = process_split("val", config.val_dir, config, logger)

    # --- 최종 요약 ---
    logger.info("")
    logger.info("=" * 70)
    train_count = len(train_records) if train_records else 0
    val_count = len(val_records) if val_records else 0
    logger.info(f"파이프라인 완료! train={train_count}, val={val_count}, total={train_count + val_count}")
    logger.info("=" * 70)


# ============================================================================
# 10. 실행 예시
# ============================================================================

if __name__ == "__main__":
    # 기본 설정으로 실행
    config = PipelineConfig(
        # --- AI허브 제공 split 경로 ---
        train_dir="./raw_data/Training",        # Training zip 해제한 폴더
        val_dir="./raw_data/Validation",        # Validation zip 해제한 폴더
        output_dir="./processed_data",
        report_dir="./reports",

        # --- 텍스트 정제: 중간 강도 ---
        clean_normalize_whitespace=True,
        clean_strip_html_tags=True,
        clean_normalize_unicode=True,
        clean_remove_control_chars=True,
        clean_collapse_newlines=2,
        clean_strip_leading_numbering=False,  # 민원 답변의 "1.", "2." 유지
        clean_remove_greeting_phrases=True,

        # --- PII: flag 모드 (먼저 리포트 확인 후 mask/drop 전환) ---
        pii_check_mask_completeness=True,
        pii_detect_phone=True,
        pii_detect_email=True,
        pii_detect_rrn=True,
        pii_detect_account=True,
        pii_detect_address=True,
        pii_action="flag",

        # --- 품질 필터링: 보수적 (데이터 최대 보존) ---
        filter_min_instruction_len=5,
        filter_min_input_len=20,
        filter_min_output_len=5,             # 요약 등 기본 태스크
        filter_max_input_len=10000,
        filter_max_output_len=5000,
        filter_drop_empty_fields=True,
        filter_min_output_word_count=2,       # 요약 등 기본 태스크
        filter_max_repetition_ratio=0.5,
        filter_drop_if_input_equals_output=True,

        # --- 단답형 output 허용 태스크 필터 ---
        qa_filter_min_output_len=1,           # "예", "문화", "민원인" 등 1글자도 허용
        qa_filter_min_output_word_count=1,    # 1어절도 허용
        qa_task_names=["질의응답", "분류"],    # 질의응답 + 분류 태스크 단답 허용

        # --- 중복 제거 ---
        dedup_exact=True,
        dedup_near=True,
        dedup_near_threshold=0.85,
        dedup_ngram_size=5,
        dedup_field="instruction_output",        # 요약/질의응답: instruction+output 기준
        dedup_near_skip_tasks=["분류"],           # 분류는 near-dup 제외 (exact만 적용)
        dedup_skip_task_field="input_output",    # 분류 exact dedup: input+output 기준 (상담 원문이 다르면 별개)

        # --- 출력 ---
        output_format="alpaca",
        include_metadata=True,
    )

    run_pipeline(config)
