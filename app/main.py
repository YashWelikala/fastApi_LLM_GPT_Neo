from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# -------------------------------
# Config
# -------------------------------
model_id = "yashWeli/energy-suggestions-gptneo-Finetune"

# Detect device
device = 0 if torch.cuda.is_available() else -1
print("Using GPU" if device == 0 else "Using CPU")

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Create pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="LLM Energy Suggestion API")


class InferenceRequest(BaseModel):
    instruction: str
    input: str
    max_tokens: int = 100


@app.post("/generate")
def generate_response(request: InferenceRequest):
    try:
        prompt = f"{request.instruction.strip()}\n{request.input.strip()}\nResponse:"
        result = text_generator(prompt, max_new_tokens=request.max_tokens)
        output_text = result[0]['generated_text'].split("Response:")[-1].strip()
        return {"output": output_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



