from fastapi import FastAPI
from pydantic import BaseModel
from prediction import Prediction

app = FastAPI()


class Message(BaseModel):
    content: str


@app.post("/")
def get_emotion(message: Message):
    tasks = ['emotion', 'hate', 'irony', 'offensive', 'sentiment-latest']

    ai = Prediction(message.content)
    for task in tasks:
        ai.get_analysis(task)

    dictionary = ai.getDictionary()
    return {
        "joy": dictionary["joy"].item(),
        "optimism": dictionary["optimism"].item(),
        "anger": dictionary["anger"].item(),
        "sadness": dictionary["sadness"].item(),
        "hate": dictionary["hate"].item(),
        "irony": dictionary["irony"].item(),
        "offensive": dictionary["offensive"].item(),
        "positive": dictionary["positive"].item(),
        "neutral": dictionary["neutral"].item(),
        "negative": dictionary["negative"].item()
    }

