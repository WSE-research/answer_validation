import json
import requests
from tqdm import tqdm

import time
import sys
sys.path.append("..")
from scripts import evaluate_answer


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
responses = json_load("../processed_data/VANILLA/qanswer_test_responses_extended-0-7000.json")
labels = json_load("../processed_data/VANILLA/qanswer_test_responses_labels.json")

is_true = json_load("../processed_data/VANILLA/is_true.json")
ids = [q['question_id'] for q in is_true]
cnt = 0
for i in tqdm(range(len(responses))):
    if responses[i]['question_id'] in ids:
        continue
    answers = list()
    if responses[i]['question_id'] != test[i]['question_id']:
        assert False
    for j in range(len(responses[i]['response'])):
        true, total = evaluate_answer.evaluate_request(
            responses[i]['response'][j]['query'],
            responses[i]['response'][j]['result'],
            test[i]['question_entity_label'],
            test[i]['question_relation']
        )
        
        if total > 0 and true/total >= 0.5:
            answers.append(True)
        else:
            answers.append(False)
        
    is_true.append({
        'question_id': responses[i]['question_id'],
        'answer_list': answers
    })
    
    if cnt%50 == 0:
        print("SAVED", cnt)
        json_save("../processed_data/VANILLA/is_true-tmp.json", is_true)
        
    cnt += 1
    
print("SAVED", cnt)
json_save("../processed_data/VANILLA/is_true-tmp.json", is_true)