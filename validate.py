# https://www.tensorflow.org/tutorials/text/transformer

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
from transformer import *

import time
import numpy as np
import matplotlib.pyplot as plt
import os, pickle
from zipfile import ZipFile

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-ec2f4b27608f.json")

from google.cloud import bigquery
from optimizer import CustomSchedule
from test import create_masks, transformer, MAX_LENGTH

client = bigquery.Client()

# Query to join on answer ID and select id, title, body from both tables
query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
    "LIMIT 1000"
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
LOAD_TOKENIZERS = True
tokenizer_q = None
tokenizer_a = None

if LOAD_TOKENIZERS:
    # print("Loading question tokenizer...")
    # tokenizer_q = pickle.load(ZipFile("./1m/tokenizer_q.zip").open("tokenizer_q.pkl", 'r'))
    # print("Loading answer tokenizer...")
    # tokenizer_a = pickle.load(ZipFile("./1m/tokenizer_a.zip").open("tokenizer_a.pkl", 'r'))
    # print("Finished loading tokenizers")

    print("Loading question tokenizer...")
    tokenizer_q = pickle.load(open("./1m/tokenizer_q.pkl", 'rb'))
    print("Loading answer tokenizer...")
    tokenizer_a = pickle.load(open("./1m/tokenizer_a.pkl", 'rb'))
    print("Finished loading tokenizers")

    sample_string = '<p>Transformer is awesome.</p>'

    tokenized_string = tokenizer_q.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))

    original_string = tokenizer_q.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))

    assert original_string == sample_string

    for ts in tokenized_string:

        print ('{} ----> {}'.format(ts, tokenizer_q.decode([ts])))

else:
    print("Set to not load tokenizers")

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_q.vocab_size + 2
target_vocab_size = tokenizer_a.vocab_size + 2
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

temp_learning_rate_schedule = CustomSchedule(d_model)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

# transformer = Transformer(num_layers, d_model, num_heads, dff,
#                           input_vocab_size, target_vocab_size, 
#                           pe_input=input_vocab_size, 
#                           pe_target=target_vocab_size,
#                           rate=dropout_rate)

def evaluate(inp_sentence):
    start_token = [tokenizer_q.vocab_size]
    end_token = [tokenizer_q.vocab_size + 1]
    
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_q.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_a.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
    
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)
        
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_a.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    
    sentence = tokenizer_q.encode(sentence)
    
    attention = tf.squeeze(attention[layer], axis=0)
    
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}
        
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
        
        ax.set_ylim(len(result)-1.5, -0.5)
            
        ax.set_xticklabels(
            ['<start>']+[tokenizer_q.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)
        
        ax.set_yticklabels([tokenizer_a.decode([i]) for i in result 
                            if i < tokenizer_a.vocab_size], 
                        fontdict=fontdict)
        
        ax.set_xlabel('Head {}'.format(head+1))
    
    plt.tight_layout()
    plt.show()

def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)
    
    predicted_sentence = tokenizer_a.decode([i for i in result 
                                                if i < tokenizer_a.vocab_size])  

    print(f'-----Input-----\n{sentence}')
    print(f'-----Predicted translation-----\n{predicted_sentence}')
    
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
    "LIMIT 1000"
)
# Executes the query
query_job = client.query(query)

# 55
for i, x in enumerate(query_job.result()):
    if i == 10:
        print(translate(x[2]))
        print(f"-----Actual-----\n{x[5]}")
        break
