import json
import re
import time
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")

qanswer_results = json_load("../processed_data/VANILLA/qanswer_test_responses.json")


qanswer_results_new = list()
cnt = 0
for q in qanswer_results[:1]:
    print("---------", q['question_id'], "-----------")

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

    qanswer_results_new.append({'question_id': q['question_id'], 'response': response})
    
    if cnt%50 == 0:
        print("SAVED", cnt)
        json_save("../processed_data/VANILLA/qanswer_test_responses_extended.json", qanswer_results_new)
        
    cnt += 1
    break
    
print("SAVED", cnt)
json_save("../processed_data/VANILLA/qanswer_test_responses_extended.json", qanswer_results_new)