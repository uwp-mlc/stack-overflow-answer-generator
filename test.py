# https://www.tensorflow.org/tutorials/text/transformer

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-666599d0246c.json")

from google.cloud import bigquery

client = bigquery.Client()

# Query to join on answer ID and select id, title, body from both tables
query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
)
# Executes the query
query_job = client.query(query)

for row in query_job.result(page_size=100):  # API request - fetches results
    # Row values can be accessed by field name or index
    # assert row[0] == row.name == row["name"]
    print(row[2])


# https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder
tokenizer_ans = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string

for ts in tokenized_string:
  print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))