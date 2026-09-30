import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.rag.prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class QwenGenerator:
    """Loads Qwen2.5 once at startup and generates answers on demand."""

    def __init__(self, model_name: str, max_new_tokens: int) -> None:
        if "qwen2.5" not in model_name.lower():
            raise ValueError(f"Expected a Qwen2.5 generation model, got: {model_name}")
        logger.info("Loading Qwen2.5 model: %s", model_name)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Qwen2.5 ready on %s", self.device)

    def generate(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        model_inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, input_ids.shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()
