# Improving the Question Answering Quality using Answer Candidate Filtering based on Natural-Language Features

## Abstract

Software with natural-language user interfaces has an ever-increasing importance.
However, the quality of the included Question Answering (QA) functionality is still not sufficient regarding the number of questions that are answered correctly.

In our work, we address the research problem of how the QA quality of a given system can be improved just by evaluating the natural-language input (i.e., the user's question) and output (i.e., the system's answer).

Our main contribution is an approach capable of identifying wrong answers provided by a QA system.
Hence, filtering incorrect answers from a list of answer candidates is leading to a highly improved QA quality. 
In particular, our approach has shown its potential while removing in many cases the majority of incorrect answers, which increases the QA quality significantly in comparison to the non-filtered output of a system.

## Approach

### The place of Answer Validation process in Knowledge Graph Question Answering
![image](https://user-images.githubusercontent.com/16652575/143687184-411a6468-2fa0-4754-8fe6-dc2faa5c8019.png)

### Our approach: to verbalize SPARQL query-candidates in order to use an Answer Validation Classifier
![image](https://user-images.githubusercontent.com/16652575/143687343-12d622ae-ddbd-4230-a3ba-9d44f2ab318b.png)

### Results for Question Answering quality improvement
![image](https://user-images.githubusercontent.com/16652575/143687377-e12788d5-7720-4e42-b48f-3d4fb9fccdf6.png)

* **P@1**: From 0.2476 to 0.4251 (i.e., by 71.7%) while using Approach 2 and to 0.2948 (i.e., by 19.1%) while using Approach 3.
* **P@5**: from 0.1036 to 0.1368 (i.e., by 32.0%) while using Approach 2 and to 0.1183 (i.e., by 14.2%) while using Approach 3.
* **NDCG@5**: from 0.3249 to 0.4698 (i.e., by 44.6%) while using Approach 2 and to 0.3787 (i.e., by 16.5%) while using Approach 3.

## Authors

[Aleksandr Gashkov](https://github.com/AlexGash)
[Aleksandr Perevalov](https://github.com/perevalov)
[Maria Eltsova](https://github.com/MariaElts)
[Andreas Both](https://github.com/anbo-de)

## Cite

To Be Done.
