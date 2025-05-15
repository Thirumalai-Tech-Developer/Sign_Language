import json

TOKEN_PATH = './token/tokenizer.json'

def tokenize(token):
    with open(TOKEN_PATH, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)
        tokenizer = tokenizer['id2label']
        for keys in tokenizer:
            if token == keys:
                return tokenizer[keys]
            return "Not Found"
