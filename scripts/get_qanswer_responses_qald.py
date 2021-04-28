import json
import time
import requests
import re
from tqdm import tqdm


headers = {
    "Content-Type": "application/json",
    'accept': 'application/json'
}

def json_load(name):
    with open(f'{name}', 'r', encoding = 'utf-8') as f:
        return json.load(f)
    
def json_save(name, item):
    with open(f'{name}', 'w', encoding = 'utf-8') as f:
        json.dump(item, f, ensure_ascii = False, indent = 2)
        
qald = json_load("../processed_data/QALD/qald_train_wdt.json")

qanswer_responses = json_load("../processed_data/QALD/qanswer_train_responses.json")
ids = [q['uid'] for q in qanswer_responses]

for q in tqdm(qald):
    if q['uid'] not in ids:
        print('================', q['uid'], '=============')
        question = q['question_text']

        response = requests.get(
            "https://qanswer-core1.univ-st-etienne.fr/api/qa/full?question={0}&lang=en&kb=wikidata".format(question)
        ).json()['queries']

        qanswer_responses.append({
            'uid': q['uid'],
            'response': [{'query': r['query'], 'confidence': r['confidence']} for r in response]
        })
        time.sleep(1)
    
json_save("../processed_data/QALD/qanswer_train_responses.json", qanswer_responses)