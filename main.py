from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import VietnameseSummarizer
from typing import Optional
import os.path

app = FastAPI(
    title="Vietnamese Text Summarizer API",
    description="API for summarizing Vietnamese text using transformer models"
)

# Initialize the summarizer at startup
summarizer = VietnameseSummarizer()
model_path = "./saved_models/vietnamese_summarizer"

# Only train if the model doesn't exist
if not os.path.exists(f"{model_path}.pkl"):
    summarizer.train_model(model_path)
    
summarizer.load_model(model_path)

class SummarizeRequest(BaseModel):
    text: str
    max_length: Optional[int] = 1000
    min_length: Optional[int] = 100

class SummarizeResponse(BaseModel):
    original_length: int
    summary: str
    summary_length: int

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    try:
        summary = summarizer.summarize(
            request.text,
            max_summary_length=request.max_length,
            min_summary_length=request.min_length
        )
        
        return SummarizeResponse(
            original_length=len(request.text),
            summary=summary,
            summary_length=len(summary)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Vietnamese Text Summarizer API. Use POST /summarize to summarize text."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)