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
        

query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/> 
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        
        SELECT ?label WHERE {{
          {uri} rdfs:label ?label .
          FILTER(LANG(?label) = "en") .
        }}
        """

qald_test_responses = json_load("../processed_data/QALD/qanswer_train_responses_extended.json")

def run_query(rq_query):
    try:
        sparql.setQuery(rq_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        return results["results"]["bindings"]
    except:
        return []
    
    
test_labels = json_load("../processed_data/QALD/qanswer_train_responses_labels.json")
ids = [q['uid'] for q in test_labels]
cnt = 0
for question in qald_test_responses:
    if question['uid'] not in ids:
        print('===============', question['uid'], '===============')
        responses = list()

        for response in question['response']:
            labels = list()
            prefix_ents = re.findall(r"(?<!\S)wd\S*:\S+", response['query'])
            no_prefix_ents = re.findall(r"<(.*?)>", response['query'])

            for ent in prefix_ents:
                results = run_query(query.format(uri=ent))
                for result in results:
                    if result["label"]["value"] not in labels:
                        labels.append(result["label"]["value"])
            for ent in no_prefix_ents:
                results = run_query(query.format(uri="<" + ent + ">"))
                for result in results:
                    if result["label"]["value"] not in labels:
                        labels.append(result["label"]["value"])

            for result in response['result']:
                for k in list(result.keys()):
                    if result[k]['type'] == 'uri':
                        q_results = run_query(query.format(uri="<" + result[k]['value'] + ">"))
                        for q_result in q_results:
                            if q_result["label"]["value"] not in labels:
                                labels.append(q_result["label"]["value"])
                    else:
                        labels.append(result[k]['value']) # type, value

            responses.append(labels)

        test_labels.append({
            'uid': question['uid'],
            'responses': responses
        })

        if cnt%5 == 0:
            print("SAVED", cnt)
            json_save("../processed_data/QALD/qanswer_train_responses_labels.json", test_labels)

        cnt += 1
    
print("SAVED", cnt)
json_save("../processed_data/QALD/qanswer_train_responses_labels.json", test_labels)