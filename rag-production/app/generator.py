import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.config import (
    MIN_NEW_TOKENS,
    MAX_NEW_TOKENS,
    DO_SAMPLE,
    NUM_BEAMS,
    TEMPERATURE,
    REPETITION_PENALTY,
    NO_REPEAT_NGRAM_SIZE,
    LENGTH_PENALTY,
    EARLY_STOPPING,
)

class Generator:
    def __init__(self, model_name: str, max_new_tokens: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, question: str, context: list[str]) -> str:
        prompt = f"""Answer using only the context.

Context:
{chr(10).join(context)}

Question:
{question}

Answer:
"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
               outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                num_beams=NUM_BEAMS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                length_penalty=LENGTH_PENALTY,
                early_stopping=EARLY_STOPPING,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating answer: {str(e)}"
