# How Textual Similarity can be Used

## The Introduction's Introduction

This is the second in a series of blog posts on textual similarity. If you haven't already read the first I recommend reading it. TL:DR that post;
1. Defines target audience
2. Caveats the series
3. Defines what textual similarity is
4. Discusses how current usage of the term is too broad

## Introduction
As we saw in the previous post, there is not a single definition of what makes two texts similar. There are different characteristics of text that matter and this 
is application dependent. I'm going to define a taxonomy of characteristics that subdivides them into 3 categories and how these characteristics matter in practice. 

**Note, I am actively updating this blog. I will change and add tasks over the next week (12/25/2023)**

## Characteristics of Text

When comparing two texts, in my opinion there are three categories of similarity. Inspired by [syntax, semantics, and pragmatics](http://www.wu.ece.ufl.edu/books/philosophy/language.html),
these categories help provide a taxonomy for characteristics of text. These are:

1. Structural, this relates to how the text looks and is written. Texts that are similar look the same.
2. Textual, this is what information is conveyed by the text. Texts that are similar convey the same information.
3. Meta-textual, this information not captured by the text. Texts that are similar can be grouped together in some way not explicitly captured by the text.

Each of these categories has multiple characteristics. Here is a tree showing some characteristics that may matter 
when comparing texts and which category they fall under.

- Structural
	- Word choice
	- Grammatical structure
- Textual
	- Tone (e.g., Sentiment, Friendliness,Formality)
	- Information being conveyed (e.g., referring to similar world events, having similar views, discussing the same thing in different ways)
		- What is being discussed?
		- How is it being discussed?
	- Style (e.g., descriptive, narrative)
- Meta-textual
	- Context in which this text exists
		- Relationship to other texts not captured by text. This could be belonging to a collection of text
		- Text Type (e.g., satire, poetry, news)
	- Purpose of the text. Not necessarily what information does it communicate, but rather what was it meant to communicate


## Some Similarity Tasks

Note that all the characteristics defined about won't necessarily be useful for a given task.
The following section will discuss what tasks that can be framed as similarity and what categories they would fall under.
It is by no means a comprehensive list of 
1. Every task that can be framed as similarity
2. What characteristics within that category apply

Each subsection below will discuss task and what categories apply to that particular task.

### Attribution and Coverage

I haven't seen much formal literature on these tasks in NLP (there may well be a name for them I'm unaware of), but [there are similar concepts in other fields like biology](https://en.wikipedia.org/wiki/Sequence_alignment).

Attribution, as I define it, is about taking a reference text and making sure that it matches an original. Basically making sure that a new text is derived
from an original or at the very least contains it without any errors. This is useful in
1. Retrieval augmented chat as a guardrail. It is important to make sure that reference materials are not being altered
2. Parsing, making sure that no mistakes were added in by the system or model. This serves as a sanity check for traditional parsing (like making sure items aren't cut off by accident) and if using prompting to write a parser it can prevent hallucinations from altering the original text


Coverage, as I define it, is a similar task where we make sure smaller subsections of text all account for a larger chunk. This might not sound useful, but naturally arises when
1. Parsing a document and making sure nothing was dropped. This serves as a sanity check for traditional parsing (like making sure items aren't dropped from a list) and if using prompting to write a parser it can prevent chunks of the text from being dropped
2. Clustering/chunking a larger text. I've seen this at work were we need 


Both these tasks are purely structural in my opinion. We only care about preserving the original text. Some aspects like the format are less
important (for example conversing the number of new lines, spacing, or maybe numbering), but relevant text should not disappear. For example if parsing a plain text list into a tree,
```
1. Here is a list
2. This is the second items
	a. A nested item
	b. Another nested item
3. A final item
```
,we would expect the output to look like
```
["Here is a list", "This is the second items", ["A nested item", "Another nested item"], "A final item"]
```

Attribution is relatively straight forward in this example. We just need to make sure every item extracted is contained in the original. Exact string matching would be 
a good use in this case.

Coverage is a bit more complicated, since some parts of the text are "missing". We did not preserve new lines nor the numbers in this case. In the apst I have used either
1. A heuristic based method that removes all matched text and uses rules to see if it is "junk text"
2. Use of approximate matching that is "junk text" aware
3. Dynamic programming and "junk text" aware methods
 
You could use a meaning aware method, but it is wholly unnecessary. It may even cause issues if the method does not account for differences that don't matter for this specific application.

<!--
### Plagiarism

The following section is inspired and draws from the taxonomy presented by [Folytnek et al.](https://dl.acm.org/doi/10.1145/3345317). Structure, meaning,
and metadata maps well to their 5 level taxonomy presenting types of plagiarism: 

1. Characters-preserving plagiarism (Structure)
• Literal plagiarism (copy and paste)
• Possibly with mentioning the source
2. Syntax-preserving plagiarism (Structure)
• Technical disguise
• Synonym substitution
3. Semantics-preserving plagiarism (Structure and meaning)
• Translation
• Paraphrase (mosaic, clause quilts)
4. Idea-preserving plagiarism (Meta-textual)
• Structural plagiarism
• Using concepts and ideas only
5. Ghostwriting (Meta-textual)


#### Structure

##### Copy and Paste/ Quilting

We only need to find exact matches over large enough spans. While for there may be some tweaking required, we can use a strategy that chunks text
and find which are similar.

##### Synonym Substitution

Let's say 
Similar Grammatical Structure. Overall still similar word 

##### quilts


Copy and paste, find exact matches
Insertion and deletion. While this 
Syntax preserving 
#### Meaning
More major rewrites

##### Synonym Substitution

Used in combination with structural simirity techniques could be used to find which strcturally similar Text

#### Meta
Textual Structure (internet historian example)
Concepts and ideas only

### Deduplication 

Deduplication, in the context of two text documents, refers to the process of identifying and removing duplicate texts or documents.
Here are two contexts in which I've used deduplication: 
1) Record cleaning (removing duplicate records in a user facing database)
2) Dataset cleaning (preprocessing for training a model)

The primary difference between those two use cases is the end use.

In the context of permanent record cleaning the end user is the person/people using that data. The cost of a false positive
is high so we may want to be rather strict about what is considered a match.Some records may look alike, but be distinct for business reasons
not readily apparent in the text.

Dataset deduplication effects so we need to understand which characteristic improve downstream performance.
While there is literature showing that deduplication improves in CASE X, CASE Y, it is not readily apparent what characteristics of text are important.
This is a question I intended to broach in a later post (ETA Summer 2024).

#### Structure

#### Meaning

#### Meta

In the context 

### Structural Similarity

These tasks mostly depend on the two texts being compared looking alike. While they may benefit from 
understanding the meaning of the text, some of the most important elements to determining similarity are purely structural.
We will motivate why that is the case for each task.

#### Deduplication of Permanent Records





#### Plagarism

Plagiarism involves finding whether one text was copied from another. There are many common tricks by plagiarists to disguise their theft:
1. Synonym substitution
2. Word insertion and deletion
3. Reordering material
4. Sentence insertion and deletion

With the exception of 1, most of these techniques can easily be detectable only using structural elements.

#### Coverage

Coverage is about verifying if one text contains one or more others. Although I've never seen this formulated as a task in of itself, there are contexts in which you will see this in NLP.

One us of this is verifying that document aggregation was done correctly. I've done this in the context of verifying annotation efforts. Part of the effort required being able to 
1) identify components of the aggregate text
2) verify all texts required were present (without mistakes)

Another use of this would be making sure that text comes from a given source. This is useful in the context of generative models if veryifying coverage source.

### Text level meaning

#### Training level deduplication 

Similarity has been used this way by multiple papers.

It's unclear what approach to deduping is best theoritically, however the approach used to dedupe falls more into this cateogry than deduplication 

#### Information Retrieval

WIP

Frankly I will not be covering IR in the depth it requires nor deserves. I just bring it up as one of the obvious and common examples of NLP tasks.

Mistmatch of text domains, limited word overlap,

#### Translation

WIP

#### Question Answering

WIP

### Metatextual meaning

This category is quite a bit broader and frankly I've only really worked with one type of task I would consider as requiring this level of information

#### Mapping

At work I often need to map regulatory texts to compliance texts. The idea is to make sure an organization has covered all the appropriate risks with procedures.

We couldn't frame this problem as one of the previous categories if we tried
1. These two texts are at completely different levels. They don't look a like, they don't use the same words. 
2. They don't even discuss the same things. The who is not the same. The risk describes what must not happen and the procedure describes what is being done without ever referencing what it prevents explicitely


A more day-to-day example of why mapping things together is hard is that I can describe a trip to the grocery store without ever using the words grocery store. Heck,
you could probably avoid any wording that would make it clear and yet still have someone understand it from context. This is hard and requires world knowledge
I have not seen any model capture. Often the solution involves hand coding or including metadata and using 
algorithms to constrain on what examples models perform analysis.

### Multi-granularity

These categories are not necessarily exclusive. You might want your method to consider multiple of these at once. How much will depend on your application and it's needs. Some applications
may also make sense at multiple of these levels. 

#### Clustering

Depending on what you want to do, clustering algorithms 
-->
## Conclusion

We've discussed and defined 3 broad categories of textual similarity and shown some examples (lol just one so far, more to come). 

Next step is to discuss how to translate these characteristics into algorithms and implement similarity in practice.