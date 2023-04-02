import csv
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax


class Prediction:
    __text = ""
    __dictionary = {}

    def __init__(self, text):
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
    def __get_labels(task):
        # download label mapping
        mapping_link = f"./ai-models/twitter-roberta-base-{task}/mapping.txt"
        with open(mapping_link, encoding="utf-8") as f:
            maps = f.read().split("\n")
            csvreader = csv.reader(maps, delimiter='\t')
        return [row[1] for row in csvreader if len(row) > 1]

    def get_analysis(self, task):
        labels = self.__get_labels(task)
        MODEL = f"ai-models/twitter-roberta-base-{task}"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)

        encoded_input = tokenizer(self.__text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        for i in range(scores.shape[0]):
            self.__dictionary[labels[i]] = scores[i]

    def getDictionary(self):
        return self.__dictionary
