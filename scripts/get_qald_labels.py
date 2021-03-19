#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

sparql = SPARQLWrapper("http://dbpedia.org/sparql")

data_name = "qald_labels.json"


# In[12]:


def open_json(file_name):
    data = None
    
    with open(file_name) as f:
        data = json.load(f)
    
    return data


# In[13]:


def save_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# In[14]:


data = open_json("qald-9-Dbpedia.json")



# In[16]:


query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?label WHERE {{
          <{uri}> rdfs:label ?label .
          FILTER(LANG(?label) = 'en') .
        }}
        """


# In[17]:


questions = dict()

for i in range(len(data)):
    print("--------", i, "--------")
    answers = list()
    question = data[i]
    for answer in tqdm(question['answers']):
        labels = list()
        if answer['DBpedia']:
            for entity in answer['DBpedia']:
                for k in list(entity.keys()):
                    if 'p' not in k and 'dbpedia' in entity[k]['value']:
                        try:
                            sparql.setQuery(query.format(uri=entity[k]['value']))
                            sparql.setReturnFormat(JSON)
                            results = sparql.query().convert()

                            for result in results["results"]["bindings"]:
                                if result["label"]["value"] not in labels:
                                    labels.append(result["label"]["value"])
                        except:
                            pass

        answers.append(labels)
        
    questions[i] = answers
    
    if i%10 == 0:
        save_json(questions, data_name)
        print("checkpoint saved", i)
        
save_json(questions, data_name)


# In[ ]:




