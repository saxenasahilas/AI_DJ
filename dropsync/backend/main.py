from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fetcher import get_stream_url
from analyzer import analyze_track
from dj_algo import find_crossfade_point, plan_drops, score_next_track

load_dotenv()

app = FastAPI()

class SearchRequest(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    stream_url: str

class CrossfadeRequest(BaseModel):
    current_track: dict
    next_track: dict

class DropsRequest(BaseModel):
    track: dict

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/search")
def search(req: SearchRequest):
    return get_stream_url(req.query)

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    return analyze_track(req.stream_url)

@app.post("/crossfade-point")
def crossfade_point(data: CrossfadeRequest):
    return find_crossfade_point(data.current_track, data.next_track)

@app.post("/plan-drops")
def drops(data: DropsRequest):
    return plan_drops(data.track)
