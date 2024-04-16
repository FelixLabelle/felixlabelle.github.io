# An In-depth Discussion of Textual Similarity: Taking a look at the toolkit, understanding our options

## The Introduction's Introduction

This is the fourth post in a series of blog posts on textual similarity. If you haven't already read the previous posts I recommend reading them. TL:DR those posts;
1. Define the target audience
2. Caveat the series
3. Define what textual similarity is
4. Discuss how current usage of the term is too broad
5. Describe characteristics of similarity that matters and provide a taxonomy to reason about them
6. Give specific examples of tasks and how they map up to the taxonomy
7. Introduce and define a taxonomy for similarity methods

## Introduction

Beyond algorithms, we need discuss considerations when choosing them.
Below is a list, in no particular order, of different considerations 

<!-- Rework order -->

1. Performance
	1. Robustness
		1. OOD
		2. noise
		3. Input size Invariance
		4. Commutativity
		5. Determinism
	2. Objective
	3. Subjective 
		1. Aligned with expectations
		2. Scores match expectations
2. Computational Requirements
	1. Training
	2. Prep Work
	3. Inference
3. Characteristics captured

Contemporary models (and most similarity methods) have three stages:
1. Training
2. Preparation
3. Inference


The following subsections will deep dive into each of the following

## Performance

### Repeatability

This section is primarily concerned with how consistent a method is. If I run it twice, do I get the same results and if I inverse argument order, do I get the same result.

Commutativity is typical in most methods, however for some of the n-gram methods it is not. For neural methods it is not either. 



### Characteristics Captured

We've mostly talked about what characteristics are, which exist. As an engineer, what matters is how well the end product works, so the discussion 
of them at length might seem a bit pointless. What matters is  whether that is how people much people "like" the end product or how well the model does some task (metrics of course). 

From an application point view, the characteristics are really just used to ground discussion about what makes two texts similar. For corpus cleaning, we only care
about removing eggregiously similar examples. So we might be interesting in methods that only check for overlap. Using word overlap won't capture synonymy, so it might be best suited for corpus cleaning.

If your application is likely to need certain types of information captured, you'll want to be sure your similarity method can capture them.

### Computational Requirements (Inference)

Time required

What hardware is used CPU vs Accelerator (GPU,TPU, etc..):

At a system level this matters. Cost, access, platform lock, etc..
Some systems can't be turned on and off at will. There are currently limits on # of gpus

A CPU can be equally fast or faster for smaller models (REFERENCE)

There are also more typical considerations such as memory usage

When there are examples stored,
retrieval time, memory usage, time to retrieve matter

### Performance

I think there are three principle components to measuring sucess for textual similarity
1. Defining the output, is it binary or is it rankings. You could use a binary model to do rankings, asumming it output some sortable output (like confidence), but there is an important distinction to be made
2. Selecting an appropriate dataset(s) to measure success. If you pick data that is wildly different, either in terms of task or domain, you're metrics are not necessarily meaningful
3. Correct metrics that capture what you are looking for. If you framed your task as a binary sucess, you're likely going to use classification metrics. For ranking other metrics like
precision @ k or NDCG make more sense.

### Input Size/Granularity Invariance 

I've noticed models have varying performance across different lengths. Try to find papers on the subject or think of a simple example

### Commutativity

Not all 
### Determinism

### Computational Requirements (Training)

Training is meant as a catch-all, this include preparing a not always necessary. Some models don't 

### Performance (Subjective)
Correlation with human judgement

### Performance (Robustness)

Does noise cause major issues? mispellt words, alternate spelling or synonyms, nicknames (Regz vs truth in lending act)

Do domain changes cause major issues?

## Conclusion

Beyond characteristics that make texts similar, there are additional considerations. 

Performance

Etc..
<!--

Considerations when selecting algorithms

1. Input order invariance
2. Order invariance (scrambling things changes the score_
3. Length invariance
4. Domain invariance
5. Characteristics captured
6. Performance on proposed task

-->

<!--
https://medium.com/@appaloosastore/string-similarity-algorithms-compared-3f7b4d12f0ff
http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html#variational
    Text overlap
		Matching Coefficient
		Dice’s Coefficient
		Jaccard Similarity or Jaccard Coefficient or Tanimoto coefficient
		Overlap Coefficient
		Regex
			GREP
			A-GREP
		ROUGE
		BLEU
		q-gram
		Compression Similarity
		
	Approximate Matching
		Minhash
		Hashlib
		Edit distance
			Levenshtein distance
			Needleman-Wunch distance or Sellers Algorithm
			Smith-Waterman distance
			Gotoh Distance or Smith-Waterman-Gotoh distance
			Monge Elkan distance
			Jaro distance metric
			Jaro Winkler
			Ukkonen Algorithms
    
	Embedding Space Based
		DISCUSSION ON DIFFERENT EMBEDDING SPACES (SoundEx distance metric)
			TF/IDF
			
		Block distance or L1 distance or City block distance
		Hamming distance
		Euclidean distance or L2 distance
		Cosine similarity
			TFIDF or TF/IDF
		BERT Score
			Element by element distances
		DTW

		
	Probability based methods
		Variational distance (KL Divergence)
		Hellinger distance or Bhattacharyya distance
		Information Radius (Jensen-Shannon divergence)
		Confusion Probability
		Harmonic Mean
		Skew divergence
		Tau
	
	Heuristic based methods
		Fellegi and Sunters (SFS) metric
		Address Matching
		DNA
			FastA
			BlastP
			Maximal matches
	
	Learned methods
		Siamese networks
		Constrastive learning
		
-->

