import json
import requests
from tqdm import tqdm
import time


api = "https://qanswer-core1.univ-st-etienne.fr/api/qa/full?question={0}&lang=en&kb=wikidata"


def json_load(name):
    with open(f'{name}', 'r', encoding = 'utf-8') as f:
        return json.load(f)
    
def json_save(name, item):
    with open(f'{name}', 'w', encoding = 'utf-8') as f:
        json.dump(item, f, ensure_ascii = False, indent = 4)
        
def read_vanilla(name):
    with open(name) as f:
        data_tmp = f.readlines()

    data = list()
    for q in data_tmp:
        data.append(json.loads(q))
        
    return data


test = read_vanilla("../data/VANILLA/Extended_Dataset_Test.json")

qanswer_test_responses = list()
cnt = 0

for q in test:
    print(f"======= {cnt} ========")
    
    question = q['question']
    try:
        response = requests.get(
            api.format(question)
        ).json()['queries']
        response = [{'query': r['query'], 'confidence': r['confidence']} for r in response]
    except:
        print("error")
        response = list()
    
    qanswer_test_responses.append({
        'question_id': q['question_id'],
        'response': response
    })
    
    if cnt%50 == 0:
        print(f"===== Saved {cnt} ======")
        json_save("../processed_data/VANILLA/qanswer_test_responses.json", qanswer_test_responses)
    
    time.sleep(1)
    cnt += 1
    
print(cnt)
json_save("../processed_data/VANILLA/qanswer_test_responses.json", qanswer_test_responses)