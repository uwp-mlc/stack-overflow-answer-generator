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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-29e738f88cfa.json")

from google.cloud import bigquery
from optimizer import CustomSchedule
from pymongo import MongoClient

client = bigquery.Client()

db_client = MongoClient('192.168.0.50', 27017)
db = db_client['tokenized_strings']
collection = db['tokenized_collection']

# Query to join on answer ID and select id, title, body from both tables
query = (
    "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
    "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
    "ON questions.accepted_answer_id = answers.id "
    "LIMIT 100"
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
LOAD_DATASETS = True
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


if LOAD_DATASETS:
    print("Loading question dataset...")
    
    if not os.path.exists('./questions.data'):
        train_q = [x["question"] for x in collection.find()]
        with open('questions.data', 'wb') as filehandle:
                    # store the data as binary data stream
            pickle.dump(train_q, filehandle)
    else:
        train_q = pickle.load(open("./questions.data", 'rb'))
    print("Loading answer dataset...")
    
    if not os.path.exists('./answers.data'):
        train_a = [x["answer"] for x in collection.find()]
        with open('answers.data', 'wb') as filehandle:
                    # store the data as binary data stream
            pickle.dump(train_a, filehandle)
    else:
        train_a = pickle.load(open("./answers.data", 'rb'))
    print("Finished loading datasets")

else:
    print("Set to not load datasets")

print(tokenizer_q.vocab_size)

MAX_LENGTH = 200

BATCH_SIZE = 24

def filter_max_length(x, y, max_length=MAX_LENGTH):
  return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)


ds_q = tf.data.Dataset.from_generator(lambda: train_q, tf.int64, output_shapes=[None])

ds_a = tf.data.Dataset.from_generator(lambda: train_a, tf.int64, output_shapes=[None])

ds = tf.data.Dataset.zip((ds_q, ds_a))
ds = ds.filter(filter_max_length).padded_batch(
    BATCH_SIZE,
    padded_shapes=([-1], [-1]))

def encode(lang1, lang2):
    # lang1 = [tokenizer_q.vocab_size] + tokenizer_q.encode(
    #     lang1.numpy()) + [tokenizer_q.vocab_size+1]

    # lang2 = [tokenizer_a.vocab_size] + tokenizer_a.encode(
    #     lang2.numpy()) + [tokenizer_a.vocab_size+1]

    lang1 = [tokenizer_q.vocab_size] + lang1 + [tokenizer_q.vocab_size+1]

    lang2 = [tokenizer_a.vocab_size] + lang2 + [tokenizer_a.vocab_size+1]
    
    return lang1, lang2

MAX_LENGTH = 40

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)

def tf_encode(q, a):
    # NOTE may need to remove [] on q and a
    return tf.py_function(encode, [a, q], [tf.int64, tf.int64])

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
        q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

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

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, 
                          pe_input=input_vocab_size, 
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

if __name__ == "__main__":
    for epoch in range(EPOCHS):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(ds):
            
            # inp, tar = tf_encode(query[2].encode(), query[5].encode())
            
            # inp and tar must be
    #         (<tf.Tensor: id=207688, shape=(64, 40), dtype=int64, numpy=
    #  array([[8214, 1259,    5, ...,    0,    0,    0],
    #         [8214,  299,   13, ...,    0,    0,    0],
    #         [8214,   59,    8, ...,    0,    0,    0],
    #         ...,
    #         [8214,   95,    3, ...,    0,    0,    0],
    #         [8214, 5157,    1, ...,    0,    0,    0],
    #         [8214, 4479, 7990, ...,    0,    0,    0]])>,
    #  <tf.Tensor: id=207689, shape=(64, 40), dtype=int64, numpy=
    #  array([[8087,   18,   12, ...,    0,    0,    0],
    #         [8087,  634,   30, ...,    0,    0,    0],
    #         [8087,   16,   13, ...,    0,    0,    0],
    #         ...,
    #         [8087,   12,   20, ...,    0,    0,    0],
    #         [8087,   17, 4981, ...,    0,    0,    0],
    #         [8087,   12, 5453, ...,    0,    0,    0]])>)
            try:
                train_step(inp, tar)
            except Exception:
                print(inp)
            
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            
        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))
            
            train_loss.reset_states()
            train_accuracy.reset_states()

            print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

