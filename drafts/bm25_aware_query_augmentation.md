# BM25 Aware Query Generation

## Background

I've been working on mapping. A friend of mine had told me about query augmentation methods. Looked like a good fit. Could make
mapping more "interpretable".

Essentially the idea is to append words to a query.

Query augmentation requires data. 

This has been in different ways like artifical generating data with QA pairs (CITE GAR) or ...

I think there are two issues with this

1. Assummes QA pairs are valid
2. Only as good as the model being used. OOD issues will cause issues

Rather than synthetically generate pairs, what if we learned augmentations directly? This does require having labelled data,
however I think this is not unusual, especially if you have a 
If there is a dataset with existing mappings, it should be possible to take the difference between the query and document,
and have a model learn those. If there is carry over, the model should learn to predict query 

## Procedure

Idea is to train a model  on mappings and use augmentations.

Train a model to predict excess words


## Sanity Check

Scores taking all words

Scores using a method to reduce irrelevant words

## Results

Not working..

Lets get metrics on what isn't working..

Metrics

1. Repetition of words
	1. Target Document
2. Words	
	1. Target document
	2. Query
	3. Training (if not using doc)
3. Distance between bm25
4. Ranking metrics 

## Conclusion

Failed atm