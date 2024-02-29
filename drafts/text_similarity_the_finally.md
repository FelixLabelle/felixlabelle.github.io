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

Even with a framework to describe how texts can be similarity, we still need to translate that into practice. This post aims to provide:
1. considerations when picking a similarity method
2. an overview of different similarity methods
3. a taxonomy with which to group them



## Method taxonomy

1. Equality
	1. Exact match
	2. Processed match
2. Text Overlap
	1. Matching Coefficient
	2. Dice’s Coefficient
	3. Jaccard Similarity or Jaccard Coefficient or Tanimoto coefficient
	4. Overlap Coefficient
	5. GREP
	6. ROUGE
	7. BLEU
	8. q-gram
	9. Compression Similarity
2. Dynamic Programming methods
	1. Edit Distance
	2. Levenshtein distance
	3. Needleman-Wunch distance or Sellers Algorithm
	4. Smith-Waterman distance
	5. Gotoh Distance or Smith-Waterman-Gotoh distance
	6. Monge Elkan distance
	7. Jaro distance metric
	8. Jaro Winkler
	9. Ukkonen Algorithms
	10. Dynamic Time Warping
	11. A-GREP
3. Approximate methods
	1. Hashlib
	2. N-gram hashing
	3. Bloom Filter
	4. LSH
4. Embedding Space Based
	1. Distance Based
		1. L1
		2. L2
		3. Cosine Similarity
	2. BERT-Score
5. Heuristic Methods
	1. Fellegi and Sunters (SFS) metric
	2. Address Matching
	3. DNA
		1. FastA
		2. BlastP
		3. Maximal matches
6. Learned methods
	1. Siamese networks
	2. Contrastive Learning

## Honorable Mention: Probabilistic Based methods
1. Variational distance (KL Divergence)
2. Hellinger distance or Bhattacharyya distance
3. Information Radius (Jensen-Shannon divergence)
4. Confusion Probability
5. Harmonic Mean
6. Skew divergence
7. Tau
	
## Conclusion


## References

https://medium.com/@appaloosastore/string-similarity-algorithms-compared-3f7b4d12f0ff
http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html#variational

<!--
https://medium.com/@appaloosastore/string-similarity-algorithms-compared-3f7b4d12f0ff
http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html#variational
    Text overlap
		1. Matching Coefficient
		2. Dice’s Coefficient
		3. Jaccard Similarity or Jaccard Coefficient or Tanimoto coefficient
		4. Overlap Coefficient
		5. Regex
			1. GREP
			2. A-GREP
		6. ROUGE
		7. BLEU
		8. q-gram
		9. Compression Similarity
		
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

