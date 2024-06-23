# Practical Tidbits: Selecting MinHash Hyper Parameters for Deduplication

In practice deduplicating large datasets is expensive and time consuming, especially if using common similarity methods like embeddings. Some embeddings are computationally less expensive to calculate and 
use like TF-IDF, but these can still be prohibitive. One  alternative is MinHash used with LSH as an index.

While there are good libraries for MinHash & LSH there are hyper parameters to pick and it's unclear which values are best for a given downstream task. [There isn't really a consensus or extensive testing as far as I'm aware, especially not for non-web domain data](https://huggingface.co/blog/dedup).

This blog presents a procedure and corresponding code to select the hyper parameters that best mimic Jaccard similarity for your data.


## Background: What are MinHash + LSH for?

To deduplicate a corpus naively we compare every text pairwise using a similarity method and flag documents that have matches above some threshold. The threshold will depend on your purpose,
for pre-training data this paper suggests ENTER VALUE.

One way of determining if two texts are similar is Jaccard similarity, using word overlap between a processed version of the texts.

The process to compute Jaccard similarity would look like:
1. For every pair of documents d_i, d_j such that i != j
	1. Tokenize the documents
	2. Word post-processing (this might not make sense in the case of deduplication since it may make two texts that are disimilar like one another)
		1. stemming/ lemmatization
		3. casing
		4. OOV/rare word handling (UNKs)
	3. Creating a set s_i and s_j (or multiset, occurrence counts)
	4. Calculate Jaccard similarity of s_i, s_k, in other words the intersection(s_i,s_k)/union(s_i,s_j)
2. Flag all pairs the threshold t

Deduplicating this way, while simple and fast compared to other methods, could be made even faster. Both steps presented above can be accelerated
1. Similarity: Set overlap (Jaccard) can be approximated using MinHash
2. Pair by pair search: We can use locality sensitive hashing (LSH) to bin together similar documents and only compare documents that may be similar


For an in depth explanation I recommend [this blog](https://giorgi.tech/blog/minhashing/) or [lessons 4](https://users.cs.utah.edu/~jeffp/DMBook/L4-Minhash.pdf) & [5](https://users.cs.utah.edu/~jeffp/DMBook/L5-LSH.pdf) of Jeff Phillip's DM:AGP course.

## Before Getting Started

This method the assumption that you are already comfortable with Jaccard similarity for your application and have a threshold for Jaccard similarity in mind. If you do, skip to the next section.

If you don't, what values of Jaccard similarity work best will depend on the [task](https://felixlabelle.github.io/2023/12/18/text-similarity-tasks.html#deduplication).
For certain tasks like system of record cleaning this might be easier to get a good gauge on through annotation or manual verification. For other tasks like pretraining dedupe to improve
performance you may need to experiment different results. In  that case you might need/want to use the method below to just pick which methods gives the precision and recall for 
a given set of hyper parameters (verses unvaried ones)

If this MinHash LSH combo is going to be used to clean up a dataset I would check impact of different settings on downstream application if possible. Ultimately don't forget this is just a step and not the end process.
Use of this method will make sure your model better follows Jacard similarity, nothing else.

Weighted min hash isn't used, 
1. I'm not familiar with its use and effects in practice
2. It can't be used online AFAICT, which the code below is setup to run as such (Although it could be modified)

## Setup

One method to determine the best hyper parameters is through experimentation, specifically how well given settings replicate the exact results (Jaccard + Item by item comp).

There are many hyper parameters that can be selected, namely
1. Hash function
2. Number of Hashes K
3. Number of Bins b
4. n-gram size(s)
5. Threshold

For each possible combination of these values the code calculates and write to JSON
1. The size in memory of the serialized index
2. Time required to index and run (speed kind of)
3. Precision, recall, F1 compared to Jaccard
4. Error compared to Jaccard

The process is as follows
0. Sample a dataset from your domain (I chose 10k, but it worked with smaller datasets as well)
1. Process the corpus into sets of n-grams
2. Calculate the true Jaccard similarity
3. For each threshold T calculate the number of matches above T
4. For each setting of K,b, T calculate the number of approximate matches
5. Calculate the precision and recall of approximate method, size required for the index, time required to compute
6. Save results
7. Pick best setting according to precision, recall, or F1 or any other criteria you have


## Code

The code uses [datasketch](https://ekzhu.com/datasketch/index.html) since it what has been used by others in the community and it works.

The code below tries different n_gram counts, hash functions, number of permutations (hash functions run), bands (LSH band size and number), as well as the threshold. You don't need to vary all of these,
more likely than not you will want to pick a threshold and n_gram_size first. Each experiments metrics and measures are stored and saved at the end.


```
# pip install tqdm
# pip install datasketch
# pip install pyfarmhash  xxhash mmh3
from itertools import product
import pickle
from time import process_time
import json

from datasketch import MinHash, MinHashLSH
from datasketch.hashfunc import sha1_hash32
import farmhash
import mmh3
import numpy as np
from tqdm.auto import tqdm
import xxhash


def mmmh3_hash(text):
    return mmh3.hash(text, signed=False)
    
def farmhash_hash(text):
    return farmhash.hash32(text.decode())
    
def xxhash_hash(text):
    text_hash = xxhash.xxh32()
    text_hash.update(text)
    return text_hash.intdigest()
    
hash_dict = {"mmh3" : mmmh3_hash,
             "farmhash" : farmhash_hash,
             "sha1" : sha1_hash32,
             "xxhash" : xxhash_hash}
             
def jaccard_similarity(set1,set2):
    try:
        return len(set1.intersection(set2))/len(set1.union(set2))
    except ZeroDivisionError:
        return 0.0

def preprocess_text(text, n_gram_size=1,pretokenized=False):
    # NOTE if # tokens < n_gram_size the set return is empty
    if pretokenized:
        tokens = text
    else:
        tokens = text.split()
    token_set = set([" ".join(tokens[i:i+n_gram_size]) for i in range(0,len(tokens)-(n_gram_size-1))])
    return token_set
    
def calculate_minhash(token_set, num_perm=128, hash_func="sha1"):
    token_set_hash = MinHash(num_perm=num_perm, hashfunc=hash_dict[hash_func])
    for token in token_set:
        token_set_hash.update(token.encode('utf-8'))
    return token_set_hash

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)   
    
def hp_gridsearch(corpus, search_params, pretokenized=False):
    results = []
    parameter_dicts = [{key: val for key,val in zip(search_params.keys(),items)} for items in product(*search_params.values())]
    jaccard_similarities = {}
    processed_text_cache = {}
    for parameter_dict in tqdm(parameter_dicts):
        n_gram_size = parameter_dict['n_gram_size']
        threshold = parameter_dict['threshold']
        if n_gram_size in processed_text_cache:
            processed_texts = processed_text_cache[n_gram_size]
        else:
            processed_texts = [preprocess_text(text, n_gram_size,pretokenized=pretokenized) for text in corpus]
            processed_text_cache[n_gram_size] = processed_texts
        if n_gram_size in jaccard_similarities:
            jaccard_similarity_matrix = jaccard_similarities[n_gram_size]
        else:
            jaccard_similarity_matrix = np.zeros((len(corpus),len(corpus)))
            indices = np.tril_indices(len(corpus))
            for i_idx,j_idx in tqdm(zip(indices[0],indices[1])):
                if i_idx == j_idx:
                    jaccard_similarity_score = 1.0
                else:
                    jaccard_similarity_score = jaccard_similarity(processed_texts[i_idx],processed_texts[j_idx])
                    jaccard_similarity_matrix[i_idx,j_idx] = jaccard_similarity_score
                # Note: we only need to set i,j once if they are equal hence this setup
                jaccard_similarity_matrix[j_idx, i_idx] = jaccard_similarity_score
            jaccard_similarities[n_gram_size] = jaccard_similarity_matrix
            
        index = MinHashLSH(threshold=threshold, num_perm=parameter_dict['num_perm'], params=(parameter_dict['num_bands'],parameter_dict['num_perm']//parameter_dict['num_bands']))
        minhash_similarities = np.ones((len(corpus),len(corpus)))*-1
        start_time = process_time()
        for idx, processed_text in tqdm(enumerate(processed_texts)):
            doc_hash = calculate_minhash(processed_text,num_perm=parameter_dict['num_perm'], hash_func=parameter_dict['hash_func'])
            result_idxs = index.query(doc_hash)
            for result_idx in result_idxs:
                result_doc_hash = calculate_minhash(processed_texts[result_idx],num_perm=parameter_dict['num_perm'], hash_func=parameter_dict['hash_func'])
                minhash_similarity = doc_hash.jaccard(result_doc_hash)
                minhash_similarities[idx, result_idx] = minhash_similarity
                minhash_similarities[result_idx, idx] = minhash_similarity
            index.insert(idx, doc_hash)
            
        end_time = process_time()
        
        # Take measures
        error_mask = np.tril(np.ones((len(corpus),len(corpus))),k=-1) > 0
        jaccard_lower_tri = jaccard_similarity_matrix[error_mask]
        minhash_lower_tri = minhash_similarities[error_mask]
        errors = jaccard_lower_tri - minhash_lower_tri
        absolute_errors = np.abs(errors)
        jaccard_matches = (jaccard_lower_tri >= threshold).sum()
        tp = ((jaccard_lower_tri >= threshold) & (minhash_lower_tri >= threshold)).sum()
        fn = ((jaccard_lower_tri >= threshold) & (minhash_lower_tri < threshold)).sum()
        fp = ((jaccard_lower_tri < threshold) & (minhash_lower_tri >= threshold)).sum()
        
        # calculate metrics
        
        try:
            precision = tp/(fp+tp)
        except ZeroDivisionError:
            precision = 0.0

        try:
            recall = tp/(fn+tp)
        except ZeroDivisionError:
            recall = 0.0

        try:
            f1 = (2*precision*recall)/ (precision + recall)
        except ZeroDivisionError:
            f1 = 0.0
        
        # log measures
        parameter_dict['index_size_bytes'] = len(pickle.dumps(index))
        parameter_dict['time_elapsed'] = end_time - start_time
        parameter_dict['fn'] = fn
        parameter_dict['tp'] = tp
        parameter_dict['fp'] = fp
        parameter_dict['jaccard_matches'] = jaccard_matches              
        # log metrics
        parameter_dict['recall'] = recall
        parameter_dict['precision'] = precision
        parameter_dict['f1'] = f1
        parameter_dict['mean_absolute_error'] = np.mean(absolute_errors)
        parameter_dict['std_absolute_error'] = np.std(absolute_errors)
        
        results.append(parameter_dict)
        
    return results
if __name__ == "__main__":
    from random import seed, sample
    import json
   
    # pip install nltk
    # NOTE: NLTK is only used for the reuters corpus, you can remove it otherwise
    import nltk
    from nltk.corpus import reuters
    
    SEED = 0
    seed(SEED)
    np.random.seed(SEED)
    nltk.download('punkt')
    
    # Only needed first time, can remove if using another corpus
    nltk.download('reuters')
    
    # HPs to vary
    search_params = {"threshold" : [0.8,0.9,0.95],
    "num_perm" : [100,200,500],
    "num_bands" : [2,5,10,20],
    "n_gram_size" : [1,2,5],
    "hash_func" : list(hash_dict.keys())}
    
    # Larger sample is exponentially slower, but more telling and likely to include matches
    
    corpus = sample(list(reuters.sents()),k=4_000)
    # NOTE: Pretokenized flag can be removed if data is not pretokenized
    results = hp_gridsearch(corpus, search_params,pretokenized=True)
    
    
    
    # Save results
    json.dump(results, open('results.json','w'),cls=NpEncoder)
```

### Potential Modifications for Your Use Case

You'll want to swap out the NLTK code for your own data. When doing that keep in mind that the data doesn't need to be pretokenized and  this can be changed by removing the "pretokenized" flag.

Note that the size of the index is specific to the method used to save it, I just used pickle to give me a general idea of the size. If you are storing it another way consider changing this measure.

Changing the HP sweep values obviously 

## Trends Observed

Things I noticed that are relevant:
1. As num_bands approaches num_perms the speed crawls to a snails pace as every single document will be compared to one another. You can avoid this by keeping values at least an order of magnitude apart.
2. In general, the choice of hash function doesn't appear to make a major difference. There are slight speed differences, but performance wise it seems relatively unimportant
3. num_perms has a point of diminishing returns performance wise and even hurts above a certain value. You might want to do multiple test runs that allow you to narrow down a range of values to test.
4. Sample size matters a lot, the higher the better. Keep in mind since matches are pairwise a higher number of samples leads to exponentially more duplicates (assuming there are any in your data), however at the cost of exponentially higher compute for GT.
5. As n_gram_sizes increase (subjectively) the matches look better, however there will be less. If you are using higher n_gram_size values consider increasing the sample size

## Closing Remarks

Above is a method of selecting hyper parameters for MinHash to select, for a given domain, what values best mimic Jaccard similarity. If you're like me and ever need to pick good HPs and can't find a reference this should be helpful.

You can pick a method based on different constraints like
1. Size in memory
2. Speed
3. Performance compared to Jaccard
4. Some combination of the above

This is likely not the most efficient way to find the best hyper parameters. You could use a better optimizer best a grid method (Bayesian optimization e,g.,), I chose a grid search I wanted to analyze the results and get a good idea of general trends for my dataset. You do you.

