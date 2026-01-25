import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Generator:
    def __init__(self, model_name: str, max_new_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate(self, question: str, context: list[str]) -> str:
        prompt = f"""Answer using only the context.

Context:
{chr(10).join(context)}

Question:
{question}

Answer:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
