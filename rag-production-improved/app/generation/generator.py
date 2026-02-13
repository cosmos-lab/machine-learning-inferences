# Import PyTorch - used for tensor operations and running the model
import torch

# Import Hugging Face classes:
# AutoTokenizer -> converts text to tokens (numbers)
# AutoModelForSeq2SeqLM -> loads encoder-decoder models (T5, BART etc.)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import generation configuration values from your app settings
# These control how the model generates output text
from app.config.settings import (
    MIN_NEW_TOKENS,        # minimum number of tokens to generate in output
    MAX_NEW_TOKENS,        # maximum number of tokens to generate in output
    DO_SAMPLE,             # whether to use sampling instead of greedy decoding
    NUM_BEAMS,             # number of beams for beam search
    TEMPERATURE,           # randomness of generation (higher = more creative)
    REPETITION_PENALTY,    # penalty to avoid repeating same words
    NO_REPEAT_NGRAM_SIZE,  # prevents repeating phrases of N words
    LENGTH_PENALTY,        # controls preference for longer/shorter output
    EARLY_STOPPING,        # stops generation when beams are finished
)


# Generator class handles loading the model + generating answers
class Generator:

    # Constructor runs once when the class object is created
    def __init__(self, model_name: str, max_new_tokens: int):

        # Decide which hardware to run model on:
        # Use GPU if available, else fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer from Hugging Face model hub
        # Converts text -> token IDs and token IDs -> text
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load Seq2Seq transformer model from Hugging Face hub
        self.model = AutoModelForSeq2SeqLM.from_pretrained(

            model_name,

            # Use half precision (float16) on GPU to reduce memory usage
            # Use full precision (float32) on CPU for compatibility
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,

            # Loads model efficiently by reducing CPU RAM usage
            low_cpu_mem_usage=True,
        )

        # Move model weights to selected device (GPU or CPU)
        self.model.to(self.device)

        # Set model to evaluation mode:
        # disables dropout and training specific layers
        self.model.eval()

        # Disable gradient calculations globally:
        # saves memory and speeds up inference
        torch.set_grad_enabled(False)

        # Store max token limit for generation
        self.max_new_tokens = max_new_tokens


    # Method used to generate an answer from context + question
    def generate(self, question: str, context: list[str]) -> str:

        # Create a structured prompt for the model
        # chr(10) inserts newline between each context entry
        prompt = f"""Answer using only the context.

Context:
{chr(10).join(context)}

Question:
{question}

Answer:
"""

        # Tokenize the prompt:
        # Converts text -> PyTorch tensors of token IDs
        # .to(self.device) moves them to GPU/CPU
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        # Run inference without gradient tracking
        with torch.inference_mode():

            # Generate output tokens using decoding strategy
            outputs = self.model.generate(

                # unpack tokenized inputs
                **inputs,

                # generation constraints
                max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=MIN_NEW_TOKENS,

                # decoding strategy parameters
                do_sample=DO_SAMPLE,
                num_beams=NUM_BEAMS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                length_penalty=LENGTH_PENALTY,
                early_stopping=EARLY_STOPPING,
            )

        # Convert generated token IDs back to readable text
        # skip_special_tokens removes tokens like <pad>, <eos>
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
