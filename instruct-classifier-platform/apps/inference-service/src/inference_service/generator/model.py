"""
apps/inference-service/src/inference_service/generator/model.py
"""
from __future__ import annotations

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class GeneratorModel:
    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, base_model: str, adapter_path: str) -> "GeneratorModel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model)

        if Path(adapter_path).exists():
            model = PeftModel.from_pretrained(base, adapter_path)
        else:
            print(f"[WARN] 어댑터 없음 ({adapter_path}) — base 모델로 동작")
            model = base

        model.to(device).eval()
        print(f"생성 모델 로드 완료 | device={device}")
        return cls(tokenizer, model, device)

    def generate(self, task: str, instruction: str, user_input: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "너는 instruction_type을 참고해서 적절한 출력을 생성하는 assistant다.",
            },
            {
                "role": "user",
                "content": f"instruction_type: {task}\ninput_text: {user_input}",
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        return result.split("instruction_type:")[0].strip()