"""
apps/inference-service/src/inference_service/classifier/model.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ClassifierModel:
    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, model_path: str) -> "ClassifierModel":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"모델 폴더를 찾을 수 없습니다: {model_path}\n"
                "Colab에서 학습한 모델을 checkpoints/ 폴더에 넣어주세요."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.to(device).eval()

        print(f"분류 모델 로드 완료 | device={device} | labels={list(model.config.id2label.values())}")
        return cls(tokenizer, model, device)

    @staticmethod
    def build_text(instruction: str, user_input: str) -> str:
        instruction = (instruction or "").strip()
        user_input = (user_input or "").strip()

        if instruction and user_input:
            return f"[INST] {instruction}\n[INPUT]\n{user_input}"
        elif instruction:
            return f"[INST] {instruction}"
        else:
            return f"[INPUT]\n{user_input}"

    def predict(self, instruction: str, user_input: str) -> dict:
        text = self.build_text(instruction, user_input)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_id = probs.argmax().item()
        pred_label = self.model.config.id2label[pred_id]
        confidence = probs[pred_id].item()

        all_scores = {
            self.model.config.id2label[i]: round(probs[i].item(), 4)
            for i in range(len(probs))
        }

        return {
            "task": pred_label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }