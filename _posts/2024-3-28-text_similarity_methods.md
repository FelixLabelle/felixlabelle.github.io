# An In-depth Discussion of Textual Similarity: Taking a look at the toolkit

## The Introduction's Introduction

This is the third post in a series of blog posts on textual similarity. If you haven't already read the previous posts I recommend reading them. TL:DR those posts;
1. Define the target audience
2. Caveat the series
3. Define what textual similarity is
4. Discuss how current usage of the term is too broad
5. Describe characteristics of similarity that matters and provide a taxonomy to reason about them
6. Give specific examples of tasks and how they map up to the taxonomy

## Introduction

Even with a framework to describe how texts can be similarity, we still need to translate that into practice. Below is a list of similarity methods I've come across.
It cannot be understated how instrumental the [SimMetrics website](http://web.archive.org/web/20081224234350/http://www.dcs.shef.ac.uk/~sam/stringmetrics.html#variational) was in this process.
I'm familiar with common GOF methods and a fair number of neural methods, but there are a large number of GOF
ML approaches that are not typically covered or mentioned otherwise.

Below is a framework grouping together methods based on how they work. This will make more sense
later when discussing how to pick between methods. The underlying mechanism limits what each method 
can do.

Considerations/how to pick between methods will be covered in the next post.

## Methods

This is an incomplete list of textual similarity methods. This list won't, and can't, cover every single method and its variants. A lot of these methods
can be tweaked or  otherwise combined to create something novel not described here. When there are too many variants to cover, notes will be left to guide the reader.

Broadly the methods are broken down to be similar in terms of implementation/underlying mechanism. The reason is that a lot of the methods grouped together, while they may seem different,
share some underlying qualities.

1. String Matching
	1. Exact match: Straight forward, are these the same strings a=b
	2. Processed match: Applying some transformation (stripping excess spaces, lower case) and check equality f(a) = f(b)
2. Overlap Matching: Process string and look at how much overlap exists between those processed values
	1. Set-overlap:
		2. Jaccard Similarity: |intersection(a,b)|/|union(a,b)|
		3. Overlap Coefficient: |intersection(a,b)|/min(|a|,|b|)
		3. MinHash
	2. N-gram overlap:
		1. [ROUGE](https://aclanthology.org/W04-1013/)
		2. [BLEU](https://aclanthology.org/P02-1040/)
		3. [q-gram](https://pdf.sciencedirectassets.com/271538/1-s2.0-S0304397500X02551/1-s2.0-0304397592901434/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIB7ceKOSffx%2Fp8%2BuOTQFVDz74BXbDMB3PEw7Nje2DbJ6AiEAh8wqLekcHK894RDxn80zMbSKG7xwtQ7fkRgoZ5f6reYqvAUI3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDO6T9kHN7v0%2FlS8onSqQBbEDONnAPWmTY48Zsg9hRTsS7lKROp1Q2ACd%2BaCVQNBWxrwyr0xNN1UW5OWrFL1W%2FhBPRDj1HCcL1E5c4imjMS6eQNZp%2BpmeMuB7c%2BnVQt4L2YeWWrsrAugbjWSjh8s7Z2GnbWg1992ZAvIICZ3qiN%2FPWajk9aySthUKoka4iJ7d91cGxkhLPJcLbGTfT%2FsFc1bfEAGog8RSRy%2BO%2FJAc69A4yVcc7P%2BvSeL2l4mNnr7fO6PbGvMKPv%2FmNnKRhxMRpsyd6bx65Hl9oEJTZcrES%2BN%2FJRjkNGhOWq8P2Z8SKZDVatrHNFNGMriZMEA4jwRRW5nUm5mgYPvly0JK6snAo7asir48Vx5T%2FTmPv%2BY1YBZwGNh8QuTT%2FEy2aHyAKQn7aCfVDnFAtX3Ik5aUM7u2qB%2FobJPQ8d6yq4P6HsTPSVkloj2xkCweCqLtnq9gaEEGgzV%2BH3X%2BAYDgyCn5Ex735BMhQM7bqgA277NkzbIvo2105lTkhfFsrQy4SX%2BfYWieox2eZ1llnRHHe068GXuQyvoSPIJXUmzS85PeOA%2FXcOgQfdrOANfSTwIwxwK%2BC33pOULK%2Bdog2m1ijCKaDvDqkAOx8hTU8HSUNTm63mFv6YsfkSCjxXXtG6SInWMljLFxA9v6J12ZyGJe3i97rYgw9D1ibh1qiSeGxHHtFsHjxpqY13FFTVlwFcaR9Qrs%2Fu5HEKLhx%2FxFsrdDAITzkM0EtEWSuv%2FFBgU5oaXZcBL8CLNAwdS4iJcOeYHVQsk5orETCff6k4P8%2BgNMHtEhi6PDlK6ah6cLKrwVIe713aTmQQMYLDosuNhILSV63XBMLgGGup5c%2FXTQhfvYojnI6iLEmdPQgroEIGPvj%2BPp04ibIrR1MLC%2Fl7AGOrEByyBrAP%2B46OcPeRFSzv5KgE%2BZCiVbJ%2F9IijdAZhp9tIH3XNfmmN8Fyxam%2F%2Br2A%2FOIi9CZwLTlVw4X%2Fu9SVwR2S8yHKwUh%2FmYJ%2FsE8AxxAbRRcftFQOdZsOt2Ot%2FX4Bfd7RTGx7329bdhzP97I0kawmf9ttlT3049vNF75Zz4N2vA8okR77BbYGesKiAuMpS8OH%2Fo26mxZH%2FsukCrQ3Zr4pRDrAuRVkgxeOG77uxsc5pOL&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240328T215302Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSDCM7K5G%2F20240328%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=810140e2e9bcb34ad7a1909041c150d5f1f8f06a34b99bf89e2d23b530563a84&hash=4b6cd03f9d56f81350d9ec6cf7efada4e2326f938256fdb718ba57a67e968840&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0304397592901434&tid=spdf-a6d51aa1-abbb-4f7b-a1f1-34b2b0b73a86&sid=8a85f97378339145551af4d-d587087fb729gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=050358500d0f055e0c5b&rr=86bad9c52b1033fa&cc=ca)
	3. Substring matching 
		1. Jaro distance metric
		2. Jaro Winkler
		3. [Ukkonen Algorithm](https://pdf.sciencedirectassets.com/271538/1-s2.0-S0304397500X02551/1-s2.0-0304397592901434/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEMb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIB7ceKOSffx%2Fp8%2BuOTQFVDz74BXbDMB3PEw7Nje2DbJ6AiEAh8wqLekcHK894RDxn80zMbSKG7xwtQ7fkRgoZ5f6reYqvAUI3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDO6T9kHN7v0%2FlS8onSqQBbEDONnAPWmTY48Zsg9hRTsS7lKROp1Q2ACd%2BaCVQNBWxrwyr0xNN1UW5OWrFL1W%2FhBPRDj1HCcL1E5c4imjMS6eQNZp%2BpmeMuB7c%2BnVQt4L2YeWWrsrAugbjWSjh8s7Z2GnbWg1992ZAvIICZ3qiN%2FPWajk9aySthUKoka4iJ7d91cGxkhLPJcLbGTfT%2FsFc1bfEAGog8RSRy%2BO%2FJAc69A4yVcc7P%2BvSeL2l4mNnr7fO6PbGvMKPv%2FmNnKRhxMRpsyd6bx65Hl9oEJTZcrES%2BN%2FJRjkNGhOWq8P2Z8SKZDVatrHNFNGMriZMEA4jwRRW5nUm5mgYPvly0JK6snAo7asir48Vx5T%2FTmPv%2BY1YBZwGNh8QuTT%2FEy2aHyAKQn7aCfVDnFAtX3Ik5aUM7u2qB%2FobJPQ8d6yq4P6HsTPSVkloj2xkCweCqLtnq9gaEEGgzV%2BH3X%2BAYDgyCn5Ex735BMhQM7bqgA277NkzbIvo2105lTkhfFsrQy4SX%2BfYWieox2eZ1llnRHHe068GXuQyvoSPIJXUmzS85PeOA%2FXcOgQfdrOANfSTwIwxwK%2BC33pOULK%2Bdog2m1ijCKaDvDqkAOx8hTU8HSUNTm63mFv6YsfkSCjxXXtG6SInWMljLFxA9v6J12ZyGJe3i97rYgw9D1ibh1qiSeGxHHtFsHjxpqY13FFTVlwFcaR9Qrs%2Fu5HEKLhx%2FxFsrdDAITzkM0EtEWSuv%2FFBgU5oaXZcBL8CLNAwdS4iJcOeYHVQsk5orETCff6k4P8%2BgNMHtEhi6PDlK6ah6cLKrwVIe713aTmQQMYLDosuNhILSV63XBMLgGGup5c%2FXTQhfvYojnI6iLEmdPQgroEIGPvj%2BPp04ibIrR1MLC%2Fl7AGOrEByyBrAP%2B46OcPeRFSzv5KgE%2BZCiVbJ%2F9IijdAZhp9tIH3XNfmmN8Fyxam%2F%2Br2A%2FOIi9CZwLTlVw4X%2Fu9SVwR2S8yHKwUh%2FmYJ%2FsE8AxxAbRRcftFQOdZsOt2Ot%2FX4Bfd7RTGx7329bdhzP97I0kawmf9ttlT3049vNF75Zz4N2vA8okR77BbYGesKiAuMpS8OH%2Fo26mxZH%2FsukCrQ3Zr4pRDrAuRVkgxeOG77uxsc5pOL&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240328T215302Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSDCM7K5G%2F20240328%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=810140e2e9bcb34ad7a1909041c150d5f1f8f06a34b99bf89e2d23b530563a84&hash=4b6cd03f9d56f81350d9ec6cf7efada4e2326f938256fdb718ba57a67e968840&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=0304397592901434&tid=spdf-a6d51aa1-abbb-4f7b-a1f1-34b2b0b73a86&sid=8a85f97378339145551af4d-d587087fb729gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=050358500d0f055e0c5b&rr=86bad9c52b1033fa&cc=ca)
		4. Compression Similarity: Using a lossless compression algorithm like gzip: |compress(a,b)|/min(|a|,|b|) range of this is [1,|a|+|b|]. Could be normalized by using d-1/(|a|+|b|-1) where d is the distance
		5. Difflib
3. Dynamic programming
	1. Edit Distance
		1. Levenshtein distance Damerau-Levenshtein distance
		2. Needleman-Wunch distance or Sellers Algorithm
		3. Smith-Waterman distance
		4. Gotoh Distance or Smith-Waterman-Gotoh distance
		5. Monge Elkan distance
	2. Dynamic Time Warping
4. Embedding Space Based (WIP, will add discussion on embeddings)
	1. L1
	2. L2
	3. Cosine Similarity
	4. BERT-Score
5. Learned methods (WIP) <!-- Add methods, losses here -->
	1. Embeddings
	2. Similarity Classification 
6. Heuristic Methods (custom problem specific approaches)

## Closing Comments
Some final remarks;
a fair number of methods to work with strings were omitted since they
1. Require a corpus (bloom filters, statistical methods (KL Divergence))
2. Don't make sense in the context of comparing two strings (GREP)

While these may make sense and can be used to compare strings, they didn't fit into how the subject has been framed so far.

Another remark is that input granularity can often be varied.
For example edit distance is often discussed at the char-level. There is nothing stopping you from doing it at
the word level instead, in fact that might make more sense for most applications. If deduplicating it is unlikely the same word will be misspelled
in one document, but correctly in another. Reducing the number of tokens will increase the speed. 

Beyond changing input granularity, methods can also be combined.
If misspelled words is a concern, you could use something like Jaccard similarity at the char level to account for any misspelt words.
Nothing is stopping you from mixing and matching techniques, it's just rarely discussed and becomes much more problem specific.
To see if a specific combination is necessary you would need metrics.

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
		
### Honorable Mentions: Probabilistic Based methods

Notably we didn't cover any statistical methods. I've mostly used these to compare corpora to one another and
while I'm sure someone somewhere has used these two compare two texts, it's not something I've seen personally.
Fo

1. Variational distance (KL Divergence)
2. Hellinger distance or Bhattacharyya distance
3. Information Radius (Jensen-Shannon divergence)
4. Confusion Probability
5. Harmonic Mean
6. Skew divergence
7. Tau
-->

