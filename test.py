# https://www.tensorflow.org/tutorials/text/transformer

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os, pickle

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-29e738f88cfa.json")

from google.cloud import bigquery

client = bigquery.Client()

# Query to join on answer ID and select id, title, body from both tables
query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
    "LIMIT 1000000"
)
# Executes the query
query_job = client.query(query)

# for row in query_job.result(page_size=100):  # API request - fetches results
#     # Row values can be accessed by field name or index
#     # assert row[0] == row.name == row["name"]
#     print(row[5])
counter = [0]
def printx(x, counter):
    counter[0] += 1
    print(counter[0])
    return x

# https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
# tokenizer_q = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (printx(row[2].encode(), counter) for row in query_job.result()), target_vocab_size=2**22)

# filehandler = open("./tokenizer_q.pkl", 'wb', pickle.HIGHEST_PROTOCOL) 
# pickle.dump(tokenizer_q, filehandler)

# counter = [0]
# tokenizer_a = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#     (printx(row[5].encode(), counter) for row in query_job.result()), target_vocab_size=2**22)

# filehandler = open("./tokenizer_a.pkl", 'wb', pickle.HIGHEST_PROTOCOL)
# pickle.dump(tokenizer_a, filehandler)

filehandler = open("./1m/tokenizer_q.pkl", 'rb', pickle.HIGHEST_PROTOCOL)
tokenizer_q = pickle.load(filehandler)

sample_string = '<p>Transformer is awesome.</p>'

tokenized_string = tokenizer_q.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_q.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:

  print ('{} ----> {}'.format(ts, tokenizer_q.decode([ts])))