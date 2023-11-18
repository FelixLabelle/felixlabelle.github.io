# Profoundly mundane, an in-depth discussion of textual similarity

In my time as an NLP practioner, I've learned that there are a lot of tools to measure similarity between two texts.
Moreover, I have them used incorrectly. For example, I worked on a project where a metric
used neural embeddings and cosine similarity to calculate similarity between generated and ground truth text. A threshold value was then used 
to establish whether or not texts were similar and was in turn used to calculate precision and recall.
However due to the, on average, relatively short text lengths this soft similarity gave misleading results compared to human judgement. 
The algorithm was overly optimistic about matching entities and gave very high precision and recall scores (~150% higher)
Switching to BLEU gave subjectively more accurate results.

Why was this even a problem in the first place The team working on the project used the best similarity method around
(according to many blogs and a paper). That's because neural embeddings didn't measure what they thought it did in this case.

In my opinion similarity is discussed in a very simplistic way. The gap which exists for people starting to use NLP is that there are no
1. no good definition of similarity
2. discussion about properties of similarity methods
3. what are the uses of similarity and what properties make sense in those contexts
4. a detailed inventory of existing similarity methods
5. a guide on how to select methods

At the end of this blog, you should be able to confidently identify what problems can be framed as 
similarity, properties your problem requires, what algorithms satisfy your contrainsts, and how to properly
measure that. This text may not be all encompassing and become outdated with time, however the underlying principle
will not change. You need to correctly identify your problem, it's requirerments, what tools are appropriate, and 
ways of evaluating success. This text aims to provide a strong base for how to do this in the context of similarity.
 
This text assumes some familiarity with NLP, ML, and similarity methods. Few if any methods will be 
explained in great detail, beyond what is necessary to illustrate their best usage. Some basic topics will be covered,
 but no more than necessary. Where necessary relevant resources will be pointed to if you'd like to learn more.

## Defining Similarity
According to Merriam-Webster, similarity is the quality or state of being similar (having characteristics in common).

Let's focus on one part of that definition, characteristics. There are different ways in which things can be similar. Consider the following four
phrases
1. Hello, how are you
2. Hey, what's up are similar
3. Bonjour, comment allez-vous
4. Hello, how are you
5. Are you up there

Let's get the obvious out of the way, 1 & 4 are not just similar, they are the same. They share all the same characteristics. Between our other
pairs, which is more similar will depend on what characteristics you are looking at. If you consider formality or meaning an important charachteristic, 1 & 3 are most similar in my opinion.
The more astute among you will have noticed 1 & 3 are in different languages. In which case, you could say 1 & 5 are more similar, inspite of having completely different meanings.
A thing all these examples share is they are questions, so in a one aspect they are all similar.

Besides pendantry, what was the point of this exercise It was to show that similarity is not this straightforward yesno exercise. There are different aspects to text and these 
can matter for your downstream application. If you'd like to group messages with similar meanings together regardless of meaning, you'd need a method than favors that criteria
over others like language. 

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

## Semantic Similarity IS NOT A SINGLE TASK

The term Semantic Similaity get's used a lot in NLP when we talk about comparing to piece of text. As of 1142023 HuggingFace states Sentence Similarity is the task of determining how similar two texts are. 

What does that even mean There are a lot of charachteristics which can make two texts similar or disimilar.

This primarily depends on the application and what matters to the downstream user. There is no singular definition and therein lies the issue.

For example, I work a lot with Control text. For those of you not familiar with RiskControl, the important bit is this

Controls are text that described procedures to mitigate risks (for example regulatory risks, i.e., making sure you don't break the rules)

This type of text has a specific writing style often contains how often it is executed, who is executing, why they are executing, etc..

An inventory of controls can contain duplicates, however when deduplicating there are many conisderations. While changes to the reasoning may not matter, if the difference between controls is the what or who,
they are likely different controls. Identifying those as duplicates would likely be bad and deleting them without supervision would wreak havoc.

An algorithm that works in the control context would not be well suited to say, deduping news text. The secret here is 

Paraphrasing would be another example of a semantic similarity task

Mapping tasks
