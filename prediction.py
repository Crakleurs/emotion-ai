import csv
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax


class Prediction:
    __text = ""
    __dictionary = {}
    __models = {
        'emotion': AutoModelForSequenceClassification.from_pretrained(f"ai-models/twitter-roberta-base-emotion"),
        'hate': AutoModelForSequenceClassification.from_pretrained(f"ai-models/twitter-roberta-base-hate"),
        'irony': AutoModelForSequenceClassification.from_pretrained(f"ai-models/twitter-roberta-base-irony"),
        'offensive': AutoModelForSequenceClassification.from_pretrained(f"ai-models/twitter-roberta-base-offensive"),
        'sentiment-latest': AutoModelForSequenceClassification.from_pretrained(f"ai-models/twitter-roberta-base"
                                                                               f"-sentiment-latest")
    }
    __tokenizers = {
        'emotion': AutoTokenizer.from_pretrained(f"ai-models/twitter-roberta-base-emotion"),
        'hate': AutoTokenizer.from_pretrained(f"ai-models/twitter-roberta-base-hate"),
        'irony': AutoTokenizer.from_pretrained(f"ai-models/twitter-roberta-base-irony"),
        'offensive': AutoTokenizer.from_pretrained(f"ai-models/twitter-roberta-base-offensive"),
        'sentiment-latest': AutoTokenizer.from_pretrained(f"ai-models/twitter-roberta-base-sentiment-latest")
    }

    def __init__(self, text: str):
        self.__text = text
        self.__dictionary = {}
        self.__preprocess()

    def __preprocess(self):
        new_text = []
        for t in self.__text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        self.__text = " ".join(new_text)

    @staticmethod
    def __get_labels(task: str):
        # download label mapping
        mapping_link = f"./ai-models/twitter-roberta-base-{task}/mapping.txt"
        with open(mapping_link, encoding="utf-8") as f:
            maps = f.read().split("\n")
            csvreader = csv.reader(maps, delimiter='\t')
        return [row[1] for row in csvreader if len(row) > 1]

    def get_analysis(self, task: str):
        labels = self.__get_labels(task)
        tokenizer = self.__tokenizers[task]
        model = self.__models[task]

        encoded_input = tokenizer(self.__text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        for i in range(scores.shape[0]):
            self.__dictionary[labels[i]] = scores[i]

    def getDictionary(self):
        return self.__dictionary
