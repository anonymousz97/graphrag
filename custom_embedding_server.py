from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Union, List
import uvicorn
from FlagEmbedding import BGEM3FlagModel

app = FastAPI()

# Load the model once when the application starts
model_name = "BAAI/bge-m3"
model = BGEM3FlagModel(model_name, use_fp16=True)
max_length = 2048

class EmbedRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    truncate: bool = True
    options: dict = None

    @validator('input', pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]

@app.post("/api/embed", response_model=EmbedResponse)
def generate_embeddings(request: EmbedRequest):
    if request.model != model_name:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not supported. Only '{model_name}' is supported.")

    inputs = request.input
    if request.truncate:
        inputs = [text[:max_length] for text in inputs]

    options = request.options or {}
    embeddings = model.encode(inputs, **options)['dense_vecs']
    return EmbedResponse(model=request.model, embeddings=embeddings.tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
