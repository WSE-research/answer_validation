import json
import re
import time
from SPARQLWrapper import SPARQLWrapper, JSON


sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")


def json_load(name):
    with open(f'{name}', 'r', encoding = 'utf-8') as f:
        return json.load(f)
    
    
def json_save(name, item):
    with open(f'{name}', 'w', encoding = 'utf-8') as f:
        json.dump(item, f, ensure_ascii = False, indent = 4)

        
qanswer_test_responses = json_load("../processed_data/QALD/qanswer_test_responses.json")

qanswer_results_new = list()
cnt = 0
for q in qanswer_test_responses:
    print("---------", q['uid'], "-----------")
    response = list()

    for r in q['response']:
        try:
            sparql.setQuery(r['query'])
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            result = results["results"]["bindings"]
            time.sleep(0.5)
        except:
            result = list()

        response.append({'query': r['query'], 'confidence': r['confidence'], 'result': result})

    qanswer_results_new.append({'uid': q['uid'], 'response': response})
    
    if cnt%50 == 0:
        print("SAVED", cnt)
        json_save("../processed_data/QALD/qanswer_test_responses_extended.json", qanswer_results_new)
        
    cnt += 1
    
print("SAVED", cnt)
json_save("../processed_data/QALD/qanswer_test_responses_extended.json", qanswer_results_new)
