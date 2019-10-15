# https://www.tensorflow.org/tutorials/text/transformer

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from bs4 import BeautifulSoup


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-5efd88711631.json")
from google.cloud import bigquery

client = bigquery.Client()

# Query to join on answer ID and select id, title, body from both tables
query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
    "LIMIT 50000"
)
# Executes the query
query_job = client.query(query)
counter = [0]
def verbose(x,counter):
    counter[0] += 1
    if counter[0] % 500 == 0:
        print(counter[0])
    return create_tokenizer(x)

def create_tokenizer(string):
    soup = BeautifulSoup(string, 'html.parser')
    return soup.get_text()


# https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
tokenizer_q = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (verbose(q[2], counter).encode() for q in query_job.result()), target_vocab_size=2**13)

counter = [0]
tokenizer_a = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (verbose(a[5], counter).encode() for a in query_job.result()), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_q.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_q.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_q.decode([ts])))