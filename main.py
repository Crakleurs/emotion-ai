from fastapi import FastAPI
from pydantic import BaseModel
from prediction import Prediction
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

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
    result = jsonable_encoder(dictionary)
    return JSONResponse(result, 200)
    # return {
    #     "joy": dictionary["joy"],
    #     "optimism": dictionary["optimism"],
    #     "anger": dictionary["anger"],
    #     "sadness": dictionary["sadness"],
    #     "hate": dictionary["hate"],
    #     "irony": dictionary["irony"],
    #     "offensive": dictionary["offensive"],
    #     "positive": dictionary["positive"],
    #     "neutral": dictionary["neutral"],
    #     "negative": dictionary["negative"]
    # }
