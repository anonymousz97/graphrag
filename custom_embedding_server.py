from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from typing import Union, List, Dict, Any
import uvicorn
from FlagEmbedding import BGEM3FlagModel

app = FastAPI()

# Load the model once when the application starts
model_name = "BAAI/bge-m3"
model = BGEM3FlagModel(model_name, use_fp16=True)

class EmbedRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    truncate: bool = True
    options: Dict[str, Any] = None

    @validator('input', pre=True)
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class EmbeddingData(BaseModel):
    object: str
    index: int
    embedding: List[float]

class UsageData(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbedResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: UsageData

@app.post("/api/embed", response_model=EmbedResponse)
def generate_embeddings(request: EmbedRequest):
    if request.model != model_name:
        raise HTTPException(status_code=400, detail=f"Model '{request.model}' not supported. Only '{model_name}' is supported.")

    inputs = request.input
    if request.truncate:
        inputs = [text[:model.max_length] for text in inputs]

    options = request.options or {}
    embeddings = model.encode(inputs, **options)['dense_vecs']

    embedding_data = [
        EmbeddingData(object="embedding", index=i, embedding=embedding.tolist())
        for i, embedding in enumerate(embeddings)
    ]

    usage_data = UsageData(prompt_tokens=len(inputs), total_tokens=len(inputs))

    response = EmbedResponse(
        object="list",
        data=embedding_data,
        model=request.model,
        usage=usage_data
    )

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=11434)
