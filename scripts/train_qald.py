#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

import os
import random
import json
import gc

# load BERT modules
from official import nlp
import official.nlp.bert as bert
import official.nlp.bert.tokenization as tokenization
import official.nlp.bert.configs as configs
import official.nlp.bert.bert_models as bert_models
import official.nlp.optimization

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle as shuffle_sklearn
from sklearn.metrics import precision_score, recall_score, f1_score

import mlflow

random_state = 42

print(f'Tensorflow version {tf.__version__}')
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# disable warning messages
tf.get_logger().setLevel('ERROR')

EXPERIMENT_NAME = 'Answer_Validation_Labels'
mlflow.set_tracking_uri("http://0.0.0.0:41250")
mlflow.set_experiment(EXPERIMENT_NAME)


# In[2]:


data_path = ['/home/aperevalov/answer_validation']


# In[3]:


def json_load(name):
    with open(f'{name}.json', 'r', encoding = 'utf-8') as f:
        return json.load(f)
    
def json_save(name, item):
    with open(f'{name}.json', 'w', encoding = 'utf-8') as f:
        json.dump(item, f, ensure_ascii = False, indent = 2)


# In[4]:


# make path from elements so it works both on windows and linux 
file_bert = ['vocab.txt']

# set up tokenizer to generate Tensorflow dataset
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(*file_bert))

print(f'Vocab size: {len(tokenizer.vocab)}')


# In[5]:


config_dict = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 30522}

bert_config = configs.BertConfig.from_dict(config_dict)


# In[6]:


# convert sentence to tokens
def encode_sentence(s):
    tokens = list(tokenizer.tokenize(s)) + ['[SEP]']
    return tokenizer.convert_tokens_to_ids(tokens)

def encode_pair(q, a, max_size):
    q_tok = ['[CLS]'] + tokenizer.tokenize(q) + ['[SEP]']
    a_tok = tokenizer.tokenize(a) + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(q_tok + a_tok)
    
    if len(ids) > max_size:
        raise IndexError('Too many tokens')
    else:
        inputs = {
            'input_word_ids': ids + [0]*(max_size - len(ids)),
            'input_mask': [1]*len(ids) + [0]*(max_size - len(ids)),
            'input_type_ids': [0]*len(q_tok) + [1]*len(a_tok) + [0]*(max_size - len(ids))
        }
        
        return inputs
    
assert(encode_sentence('Human is instance of animal') == [2529, 2003, 6013, 1997, 4111, 102])
assert(
    encode_pair('Who are you?', 'I am your dad.', 15)['input_word_ids'] ==
    [101, 2040, 2024, 2017, 1029, 102, 1045, 2572, 2115, 3611, 1012, 102, 0, 0, 0]
)


# In[7]:


def bert_encode(class_0, class_1, tokenizer, size=0, random_state=42):
    # random.shuffle(class_0[:size] if size else class_0)
    # random.shuffle(class_1[:size] if size else class_1)
    
    labels = [0]*len(class_0) + [1]*len(class_1)
    records = class_0 + class_1

    questions = tf.ragged.constant([encode_sentence(s[0]) for s in records])
    answers = tf.ragged.constant([encode_sentence(s[1]) for s in records])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*questions.shape[0]
    input_word_ids = tf.concat([cls, questions, answers], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_question = tf.zeros_like(questions)
    type_answer = tf.ones_like(answers)
    input_type_ids = tf.concat([type_cls, type_question, type_answer], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs, tf.convert_to_tensor(labels)


# In[8]:


class DataGenerator(tf.keras.utils.Sequence):
    'generates data batches'
    def __init__(self, class_0, class_1, embed_len=180, batch_size=32, shuffle=True, random_state=42):
        'Initialization'
        self.text = class_0 + class_1
        self.labels = [0]*len(class_0) + [1]*len(class_1)
        self.embed_len = embed_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = len(self.text)
        self.indexes = list(range(self.count))
        self.data = [None]*len(self.text)
        self.on_epoch_end()
        
        if self.shuffle:
            self.indexes = shuffle_sklearn(self.indexes, random_state=random_state)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.count // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        inputs = {
            'input_word_ids': [],
            'input_mask': [],
            'input_type_ids': []}
        
        outputs = []
        
        for i in indexes:
            if not self.data[i]:
                self.data[i] = encode_pair(self.text[i][0], self.text[i][1], self.embed_len)
            for key in inputs:
                inputs[key] += [self.data[i][key]]
            outputs.append(self.labels[i])
            
        for key in inputs:
            inputs[key] = tf.ragged.constant(inputs[key], inner_shape=(self.batch_size, self.embed_len))
        outputs = tf.convert_to_tensor(outputs)

        return inputs, outputs
    
    def get_dataset(self):
        inputs = {
            'input_word_ids': [],
            'input_mask': [],
            'input_type_ids': []}
        
        outputs = []
        
        for i in range(len(self.text)):
            if not self.data[i]:
                self.data[i] = encode_pair(self.text[i][0], self.text[i][1], self.embed_len)
            for key in inputs:
                inputs[key] += [self.data[i][key]]
            outputs.append(self.labels[i])
            
        return inputs, outputs


# In[9]:


# story batch + epoch results and dump to file if name is not empty
class HistoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, file_name, history={'epoch': [], 'batch': []}):
        self.history = history
        self.name = file_name
        self.epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        logs['epoch'] = self.epoch
        self.history['epoch'].append(logs)
        
        if self.name:
            json_save(self.name, self.history)

    def on_train_batch_end(self, batch, logs=None):
        if logs and batch:
            logs['batch'] = batch
            logs['epoch'] = self.epoch
            self.history['batch'].append(logs)


# In[10]:


# create dataset for textual representation
"""def create_train_set_samples(data):
    class_0 = []
    class_1 = []
    
    for k, v in data.items():
        if v:
            for i in v['generated']['right']:
                class_1.append([v['vanilla']['question'], i['text'], 1])
            for i in v['generated']['wrong']:
                class_0.append([v['vanilla']['question'], i['text'], 0])

    return class_0, class_1, 'quanswer'"""


# In[11]:


def create_train_set_samples_qald(data):
    class_0 = list()
    class_1 = list()

    for i in range(len(data)):
        true_answers = get_list_of_true_answers(qald_dataset[i]['answers']) # from original dataset

        for j in range(len(qald[i]['answers'])): # data provided by AlGa
            is_answer_correct = if_answer_correct(true_answers, qald[i]['answers'][j])
            if is_answer_correct:
                if len(data[str(i)][j]) > 0 and len(' '.join(data[str(i)][j]).split()) < 75:
                    class_1.append((find_english_in_qald(qald_dataset[i]['question']), ' '.join(data[str(i)][j]))) # my data
            else:
                if len(data[str(i)][j]) > 0 and len(' '.join(data[str(i)][j]).split()) < 75:
                    class_0.append((find_english_in_qald(qald_dataset[i]['question']), ' '.join(data[str(i)][j]))) # my data
                # TODO: search for textual answer
    
    return class_0, class_1, 'labels_qald'


# In[12]:


# data_file = data_path + ['vanilla_qanswer_results']
# data = json_load(os.path.join(os.sep, *data_file))


# In[13]:


qald = json_load("qald-9-Dbpedia")
data = json_load("qald_labels")

qald_test = json_load("qald-9-test-multilingual")
qald_train = json_load("qald-9-train-multilingual")

qald_dataset = qald_train['questions'] + qald_test['questions']


# In[14]:


def find_english_in_qald(representations):
    """
    representations: qald_dataset[i]['question']
    """
    for r in representations:
        if r['language'] == 'en':
            return r['string']
    
    assert False


# In[15]:


def get_list_of_true_answers(qald_answers):
    """
    qald_answers: qald_train['questions'][0]['answers']
    """
    true = list()
    if 'bindings' in qald_answers[0]['results']:
        for bind in qald_answers[0]['results']['bindings']:
            k = list(bind.keys())[0]
            true.append(bind[k]['value'])
    
    return true


# In[16]:


def if_answer_correct(true_uri, pred_answer):
    """
    pred_answers: qald[i]['answers'][j]
    """
    uris = list()
    if pred_answer['DBpedia']:
        for triple in pred_answer['DBpedia']:
            for k in list(triple.keys()):
                if 'p' not in k and 'dbpedia' in triple[k]['value']:
                    uris.append(triple[k]['value'])
    # answer_uris.append(uris)
    # print(true_uri)
    if any(true in uris for true in true_uri):
        return True
    else:
        return False


# In[17]:


# quick consistence check
for i in range(len(qald)):
    assert len(qald[i]['answers']) == len(data[str(i)])


# In[18]:


class_0, class_1, dataset_name = create_train_set_samples_qald(data)
print(dataset_name)
print(len(class_0))
print(len(class_1))


# In[19]:


class_1[50], class_0[50]


# In[20]:


file_pretrained = ['bert_classifier.h5']

def create_model(bert_classifier, epochs=100, batch_size=8, batches_per_epoch=1000, warmup_epochs=5):
    num_train_steps = epochs*batch_size*batches_per_epoch
    warmup_steps = batches_per_epoch*warmup_epochs

    # creates an optimizer with learning rate schedule
    optimizer = nlp.optimization.create_optimizer(
        2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    bert_classifier.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics)
    
    bert_classifier.load_weights(os.path.join(*file_pretrained))
    
    return bert_classifier


# In[21]:


def train_model(bert_classifier, 
                class_0, class_1, 
                valid_0, valid_1, 
                name,
                epochs=5, 
                batch_size=8, 
                warmup_epochs=5,
                embed=96,
                rand_idx=0 # use different random index to shuffle training data
               ):
    
    
    rs = list(range(2010, 2020)) # generate random states
    
    valid_set, valid_labels = bert_encode(valid_0, valid_1, tokenizer) 
    train = DataGenerator(class_0, class_1, embed_len=embed, batch_size=batch_size, random_state=rs[rand_idx])
    n_steps = len(class_0 + class_1) // batch_size # define number of steps per epoch
    
    history = HistoryCallback(file_name=name, history={'epoch': [], 'batch': []})
    
    bert_classifier.fit(
        train,
        steps_per_epoch=n_steps,
        validation_data=(valid_set, valid_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[
            history,
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                             mode='max',
                                             patience=1,
                                             restore_best_weights=True)
        ],
    )    


# In[22]:


train_0, valid_0 = train_test_split(class_0, test_size=0.1, random_state=4)
train_1, valid_1 = train_test_split(class_1, test_size=0.1, random_state=4)

train_0, test_0 = train_test_split(train_0, test_size=0.33, random_state=4)
train_1, test_1 = train_test_split(train_1, test_size=0.33, random_state=4)

test_set, test_labels = bert_encode(test_0, test_1, tokenizer)
# print(f"train size: {len(train_0)}", f"val size: {len(valid_0)}", f"test size: {len(test_0)}")


# In[23]:


for i in range(10):
    gc.collect()
    
    batch_size = 16
    embed = 256
    
    bert_classifier, bert_encoder = bert_models.classifier_model(bert_config, num_labels=2)
    bert_classifier = create_model(bert_classifier, batch_size=8)

    train_model(bert_classifier, 
                class_0, class_1,
                valid_0, valid_1, 'generated.json',
                batch_size=batch_size, embed=embed, rand_idx=i)
    
    y_pred = bert_classifier.predict(test_set, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = test_labels.numpy()
    
    with mlflow.start_run():
        mlflow.log_param("Rand idx", i)
        mlflow.log_param("Batch Size", batch_size)
        mlflow.log_param("Embed", embed)
        mlflow.log_metric("Accuracy", accuracy_score(y_true, y_pred))
        mlflow.log_metric("Precision", precision_score(y_true, y_pred))
        mlflow.log_metric("Recall", recall_score(y_true, y_pred))
        mlflow.log_metric("F1 Score", f1_score(y_true, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




