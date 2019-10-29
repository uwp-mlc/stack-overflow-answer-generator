import pickle, os
from google.cloud import bigquery
from multiprocessing import Pool



def work(q, a, tokenizer_q, tokenizer_a):
    return tokenizer_q.encode(q), tokenizer_a.encode(a)

if __name__ == "__main__":
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
        tokenizer_q = pickle.load(open("./200k/tokenizer_q.pkl", 'rb'))
        print("Loading answer tokenizer...")
        tokenizer_a = pickle.load(open("./200k/tokenizer_a.pkl", 'rb'))
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

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("./My Project 63888-29e738f88cfa.json")

    query = (
        "SELECT questions.id as `q_id`, questions.title as `q_title`, questions.body as `q_body`, answers.id as `a_id`, answers.title as `a_title`, answers.body as `a_body` FROM `bigquery-public-data.stackoverflow.posts_questions` AS `questions` "
        "INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS `answers` "
        "ON questions.accepted_answer_id = answers.id "
        "LIMIT 1000000"
    )

    client = bigquery.Client()

    query_job = client.query(query)

    count = 0

    questions = []
    answers = []
    
    pool = Pool(processes=10)  # lets use just 2 workers
    queue = []  # a queue for our current worker async results, a deque would be faster

    for q in query_job.result():
        queue.append(pool.apply_async(work, [q[2].encode(), q[5].encode(), tokenizer_q, tokenizer_a]))
        count += 1
        if count % 1000 == 0:
            print(count)
        while len(queue) >= pool._processes:
            process = queue.pop(0)  # grab a process response from the top
            process.wait(0.1)  # let it breathe a little, 100ms should be enough
            if not process.ready():  # a sub-process has not finished execution
                queue.append(process)  # add it back to the queue
            else:
                q, a = process.get()
                questions.append(q)
                answers.append(a)

    while len(queue) > 0:
        process = queue.pop(0)  # grab a process response from the top
        process.wait(0.1)  # let it breathe a little, 100ms should be enough
        if not process.ready():  # a sub-process has not finished execution
            queue.append(process)  # add it back to the queue
        else:
            q, a = process.get()
            questions.append(q)
            answers.append(a)

    with open('questions.data', 'wb') as filehandle:
                # store the data as binary data stream
                pickle.dump(questions, filehandle)

    with open('answers.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(answers, filehandle)
    pool.close()

    