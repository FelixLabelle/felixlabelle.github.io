# An In-depth Discussion of Textual Similarity: Taking a look at the toolkit, understanding our options

## The Introduction's Introduction

This is the third post in a series of blog posts on textual similarity. If you haven't already read the previous posts I recommend reading them. TL:DR those posts;
1. Define the target audience
2. Caveat the series
3. Define what textual similarity is
4. Discuss how current usage of the term is too broad
5. Describe characteristics of similarity that matters and provide a taxonomy to reason about them
6. Give specific examples of tasks and how they map up to the taxonomy

## Introduction

We need to define what is being returned and how to measure success in similarity.


## Features by which similarity methods

1. Granularity
2. Speed
3. Correctness
4. characteristics captured
5. Order invariance
6. Reversibility f(a,b) = f(b,a)
7. Randomness
8. Overlap sensitivtiy
9. Score vs Binary

## Types of returns
While there may well be other types of returns, I'm just familiar with two, binary and continuous. These are not necessarily mutually exclusive
### Binary

### Continous

## Types of comparisons

### Pairwise

### Corpus Aware

## 

## Evaluating similarity

There are different ways of evaluating them and they will reflect the need of the application:

Regardless of the framing you should first develop a benchmark and this requires a dataset and metrics. Do

### Binary Similarity


Are they the same or not using a threshold. This is similar to any other binary similarity task

### Ranking Similarity

IR requires this 

Are the distances measured meaningful (matching human judgement or requirements)? Are they monotone?

Some benchmarks I've seen in IR are good candidates for this.


## Conclusion

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

