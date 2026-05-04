from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fetcher import get_stream_url
from analyzer import analyze_track

load_dotenv()

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    stream_url: str

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/search")
def search(req: SearchRequest):
    return get_stream_url(req.query)

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return analyze_track(req.stream_url)
