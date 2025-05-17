import json


def tokenize(token, token_path):
    with open(token_path, 'r', encoding='utf-8') as f:
        tokenizer = json.load(f)
    tokenizer = tokenizer['id2label']
    for keys in tokenizer:
        if str(token) == keys:
            return tokenizer[keys]
    return "Not Found"