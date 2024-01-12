from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Sentences(BaseModel):
    sentence1: str
    sentence2: str


@app.post("/checkparaphrase")
async def checkparaphrase(sentences: Sentences):
    paraphrase_result = f"{sentences.sentence1} {sentences.sentence2}"
    similarity = pd.getSimilarity(sentences.sentence1, sentences.sentence2)
    return {"parphrase_probability": similarity[0], "paraphrase_result": similarity[1]}
