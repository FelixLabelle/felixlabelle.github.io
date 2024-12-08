# Practical Tidbits: ElasticSearch with custom Embeddings (Vectors) for Versions Greater than 7.6

Recently I deployed embeddings on ElasticSearch for [mapping together two types of documents](https://felixlabelle.github.io/2023/12/18/text-similarity-tasks.html#mapping) [as part of a platform we're developing for work](https://www.pwc.com/us/en/products/risk-link.html). The reasons were
1. We already had an instance of ES deployed
2. There are existing data pipelines :)
3. After some initial reading/research it looked relatively straightforward

There are two pieces of code I used as references. Both use ElasticSearch client to upload and query with custom embeddings
1. [This blog post by Bachir (unsure about his last name) which uses SentenceTransformers](https://dzlab.github.io/nlp/2021/08/09/elasticsearch_bert/)
2. [An example from ElasticSearch's lab team which uses TensorFlow](https://github.com/jtibshirani/text-embeddings)

I wrote code heavily inspired from these two sources, however every time I would run the code I would get the following error:

```
RequestError: RequestError(400, 'search_phase_execution_exception', 'runtime error')
```

Note that there can be multiple causes for this and you should look in the logs.

After some research and experimenting, turns out there was a [breaking change in ES' script in version 7.6 of ES to how vector functions are written](https://www.elastic.co/guide/en/elasticsearch/reference/7.17/breaking-changes-7.6.html#_update_to_vector_function_signatures).  The only difference is that
'doc' is no longer references in the search query (see example below for the difference).

The changes required are minor (literally 5 chars), but this makes some of the top google search results for how to do vector search in ES out of date. Which is frustrating if you are just looking for a quick way of deploying embeddings into ES.

For that reason here is an updated version of [Bachir's code](https://dzlab.github.io/nlp/2021/08/09/elasticsearch_bert/) that will work *for ES versions greater than 7.6*. If using an earlier version I would highly recommend the sources above, they are excellent.

```
from elasticsearch import Elasticsearch

INDEX_NAME = "test-index"
COLUMN_NAME = "embeddings"

# Read in and process data
f = open('data.json',)
documents = json.load(f)
corpus = []
for doc in documents:
    text = doc['text']
    embeddings = model.encode(text)
    doc[COLUMN_NAME] = embeddings.tolist()

# TODO: INSERT YOUR URL AND CREDS HERE
es = Elasticsearch()

# NOTE:  If writing to an existing index like I was
#  I recommend using scan, iterating through data, embedding it, and adding back the embeddings as a new column
for idx, doc in enumerate(documents):
    res = es.index(index=INDEX_NAME, id=idx+1, body=doc)

# TODO: Replace with your query or logic to read in queries
query = "Here is an example query"
query_vector = model.encode(text).tolist()


script_query = {
    "script_score": {
        "query": {"match_all": {}},
        "script": {
            "source": f"cosineSimilarity(params.query_vector, '{COLUMN_NAME}') + 1.0", # NOTE: Change made for versions > 7.6 is doc['{COLUMN_NAME}'] -> '{COLUMN_NAME}'
            "params": {query_vector: query_vector}
        }
    }
}
search_body = {
    "size": 10,
    "query": script_query,
    "_source": {"excludes": [COLUMN_NAME]}
}

# NOTE: Call
result = es.search(index=INDEX_NAME, body=search_body)

```
