# An In-depth Discussion of Textual Similarity: Starting the Conversation

## Target Audience

This series of blog posts assumes some familiarity with NLP, ML, and textual similarity methods. Few if any methods will be 
explained in great detail, beyond what is necessary to illustrate their best usage. Some basic topics will be covered,
but no more than necessary. Where relevant, additional resources will be pointed to if you'd like to learn more.
 
## Introduction

In my time using Natural Language Processing (NLP), I've learned that there are a lot of tools to measure similarity between two texts.
Moreover, I have them used incorrectly. For example, I worked on a project where a metric
used neural embeddings and cosine similarity to calculate similarity between generated and ground truth text. A threshold value was then used 
to establish whether or not texts were similar and was in turn used to calculate precision and recall.
However due to the, on average, relatively short text lengths this soft similarity gave misleading results compared to human judgement. 
The algorithm was overly optimistic about matching entities and gave very high precision and recall scores (~150% higher)
Switching to BLEU gave subjectively more accurate results. Not to say it was perfect, but it was more conversative which makes sense in this context.

Why was this even a problem in the first place? The team working on the project used the best similarity method around
(according to many blogs and a paper). That's because neural embeddings didn't measure what they thought it did in this case.

This type of issue is not uncommon. At work we've had a handful of projects which use similarity and every single one utilized different methods. The methods used were not always good fits.
In my opinion the issues stems from the fact that there is no:
1. good definition of similarity
2. discussion around what are the use cases for similarity and what properties make sense in those contexts 
3. detailed inventory of existing similarity methods
4. list of the desired properties of similarity methods
5. guide on how to select methods


This first post will address issue 1. and future blog posts will discuss the other 4 points. At the end of this series, you should be able to confidently identify what problems can be framed as 
similarity, properties your problem requires, what algorithms satisfy your contrainsts, and how to properly
measure that. This text may not be all encompassing and become outdated with time, however the underlying principle
will not change. You need to correctly identify your problem, it's requirerments, what tools are appropriate, and 
ways of evaluating success. This text aims to provide a strong base for how to do this in the context of similarity.

**Note, I am actively updating this post. It probably has mistakes, of all sorts, and will likely change over the next week (11/25/2023)**

## Defining Similarity
According to Merriam-Webster, similarity is the quality or state of being similar (having characteristics in common).

Let's focus on one part of that definition, characteristics. There are different ways in which things can be similar. Consider the following 5 phrases
1. Hello, how are you
2. Hey, what's up
3. Bonjour, comment allez-vous
4. Hello, how are you
5. Are you up there

Let's get the obvious out of the way, 1 & 4 are not just similar, they are the same. They share all the same characteristics. Between our other
pairs, which is more similar will depend on what characteristics you are looking at. If you consider formality or meaning an important charachteristic, 1 & 3 are most similar in my opinion.
The more astute among you will have noticed 1 & 3 are in different languages. In which case, you could say 1 & 5 are more similar, inspite of having completely different meanings.
A thing all these examples share is they are questions, so in a one aspect they are all similar.

Besides pendantry, what was the point of this exercise It was to show that similarity is not this straightforward yes/no exercise. There are different dimensions to text and these 
can matter for your downstream application. If you'd like to group messages with similar meanings together regardless of meaning, you'd need a method than favors that criteria
over others like language. 

## Charachteristics of Text
Below are some examples of charachteristics that could be used to find similar texts

- Syntax
	- Language being used
	- Word choice
	- Granularity of text
	- Grammatical structure
		- Shared stuctue
	- Length
- Semantics
	- Object
	- Subject
	- Action
	- Facts
- Pragmatics
	- Tone (e.g., Sentiment, Friendliness,Formality)
	- Context (e.g., refering to similar world events, having similar views, discussing the same thing in different ways)
	- Text Type (e.g., satire, poetry, news)
	- Style (e.g., descriptive, narrative)
	
Keep these in mind as we discuss different tasks.

## PSA: Semantic Similarity is not a (Single) Task

The term Semantic Similarity get's used a lot in NLP when textual similarity is brought up. As of 11/04/2023 HuggingFace states Sentence Similarity is "the task of determining how similar two texts are". 

What does that even mean?? As we just saw there are a lot of charachteristics which can make two texts similar or disimilar.

This primarily depends on the application and what matters to the downstream user. There is no singular definition and therein lies the issue.

Let's say two texts discuss the same subject, but have different tones is that similar? It might be if you are deduplicating a dataset for training, but if you are comparing news articles for how a subject is covered probably not.

Identifying what matter for a task is a critical part of the development process. Due to the complexity of the subject we'll save that for another blog post.

## Next Time on Semantic Similarity

So we've covered why I think similarity is an under explored subject and come up with a working definition of it. The next step is to see that in practice. The next blog post
will discuss use of similarity in tasks. What tasks can be framed as a similarity problem and some examples of what characteristics matter in each use case.