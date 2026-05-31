from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


SYSTEM_PROMPT = """Sen Türk hukuku için çalışan dikkatli bir RAG asistanısın.
Cevabı yalnızca verilen bağlamlara dayanarak üret.
Bağlamda açıkça bulunmayan bilgileri uydurma.
Cevap kısa, doğrudan ve Türkçe olmalı."""


def build_rag_prompt(question: str, contexts: List[Dict[str, object]]) -> str:
    context_blocks = []

    for ctx in contexts:
        context_blocks.append(
            f"[Bağlam {ctx['rank']} | Dosya: {ctx['file_name']} | Chunk: {ctx['chunk_id']}]\n"
            f"{ctx['text']}"
        )

    joined_contexts = "\n\n".join(context_blocks)

    prompt = f"""<s>[INST] {SYSTEM_PROMPT}

Bağlamlar:
{joined_contexts}

Soru:
{question}

Cevap: [/INST]"""

    return prompt


class RAGGenerator:
    def __init__(
        self,
        base_model_name: str,
        adapter_id_or_path: Optional[str] = None,
        max_input_length: int = 2048,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
        do_sample: bool = False,
        use_4bit: bool = True,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_id_or_path = adapter_id_or_path
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        model_kwargs = {}

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.float16

            if use_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs,
        )

        if adapter_id_or_path:
            if PeftModel is None:
                raise ImportError(
                    "peft is required for loading fine-tuned adapters. "
                    "Install it with: pip install peft"
                )

            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_id_or_path,
            )

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        if torch.cuda.is_available():
            inputs = inputs.to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        new_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return answer.strip()