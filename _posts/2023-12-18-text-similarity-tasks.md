# An In-depth Discussion of Textual Similarity: Characteristics and When They Matter

## The Introduction's Introduction

This is the second in a series of blog posts on textual similarity. If you haven't already read the first I recommend reading it. TL:DR that post;
1. Defines target audience
2. Caveats the series
3. Defines what textual similarity is
4. Discusses how current usage of the term is too broad

## Introduction
As we saw in the previous post, there is not a single definition of what makes two texts similar. There are different characteristics of text that matter and this 
is application dependent. While we introduced the problem and concept of characteristics, it doesn't give us a framework with which to think when approaching a problem that
can be framed as similarity.

## Characteristics of Text

When comparing two texts, in my opinion there are three categories of similarity. These are:

1. Structural, this relates to how the text looks and is written. Texts that are similar look the same.
2. Textual, this is what information is conveyed by the text. Texts that are similar convey the same information.
3. Meta-textual, this information not captured by the text. Texts that are similar can be grouped together in some way not explicitly captured by the text. This could be style, similar concepts, etc..

Inspired by [syntax, semantics, and pragmatics](http://www.wu.ece.ufl.edu/books/philosophy/language.html), these categories help provide a taxonomy for characteristics of text. They help differentiate 
matches based on surface level characteristics from textual meaning or contextual elements.

Each of these categories is comprised of characteristics.

- Syntactic
	- Word choice
	- Grammatical structure
- Textual
	- Information being conveyed (e.g., referring to similar world events, having similar views, discussing the same thing in different ways). I typically think of this as being comprised of the "5Ws":
		- Who: What parties (people, organizations) are involved
		- Where: Is this occurring in a specific place? This doesn't need to be a physical place, it could be a forum or even a context
		- When: Is there a temporal element to the information, like a specific time or is it recurrent?
		- What: The who is doing what?
		- Why: Are there motivations or reasons for doing this
- Meta-textual
	- Context in which this text exists
		- Relationship to other texts not captured by text. This could be belonging to a collection of text
		- Text Type (e.g., satire, poetry, news)
	- Purpose of the text. Not necessarily what information does it communicate, but rather what was it meant to communicate
	- Nature of the text
		- Tone (e.g., Sentiment, Friendliness,Formality)
		- Style (e.g., descriptive, narrative)


## Some Similarity Tasks

Not all the characteristics defined will be useful for a given task nor is the list above comprehensive.
The following section will discuss what tasks that can be framed as similarity and what categories they would fall under.
It is by no means a comprehensive list of 
1. Every task that can be framed as similarity
2. What characteristics within that category apply

Each subsection below will discuss task and what categories apply to that particular task.

### Attribution and Coverage

I haven't seen much formal literature on these tasks in NLP (there may well be a name for them I'm unaware of), but [there are related concepts in other fields like biology](https://en.wikipedia.org/wiki/Sequence_alignment).

Attribution, as I define it, is about taking a reference text and making sure that it matches an original. Basically making sure that a new text is derived
from an original or at the very least contains it without any errors. This is useful in
1. Retrieval augmented chat as a guardrail. It is important to make sure that reference materials are not being altered
2. Parsing, making sure that no mistakes were added in by the system or model. This serves as a sanity check for traditional parsing (like making sure items aren't cut off by accident) and if using prompting to write a parser it can prevent hallucinations from altering the original text


Coverage, as I define it, is a related task where we make sure smaller subsections of text all account for a larger chunk. This might not sound useful, but naturally arises when parsing a document and making sure nothing was dropped. Coverage serves as a sanity check for traditional parsing (like making sure items aren't dropped from a list) and if using prompting to write a parser it can prevent chunks of the text from being dropped.


#### The structural side of attribution

There are two use cases I have in mind: parsing and retrieval augmented generation (RAG). The following section will discuss parsing, but the principles apply to both. RAG will 
be further mentionned in the following subsection.

While parsing documents is not something all NLP practitioner have done or will need to do, I have done that several times in different use cases. Often when working with large
legal texts they have a natural structure that subdivides them into smaller sections. People work at this level and we need to subdivide the text for it to be useful for SMEs and 
downstream models are often trained with this finer grained data rather whole regulations.
Parsing legal text is not always easy and to verify the accuracy and completeness of our parse

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

Coverage is a bit more complicated, since some parts of the text are "missing". We did not preserve new lines nor the numbers in this case. In the past I have used either
1. A heuristic based method that removes all matched text and uses rules to see if it is "junk text"
2. Use of approximate matching that is "junk text" aware
3. Dynamic programming and "junk text" aware methods
 
You could use a meaning aware method, but it is wholly unnecessary. It may even cause issues if the method does not account for differences that don't matter for this specific application.

#### When structure isn't enough 

In the case of RAG, rephrasing documents is natural or even desired. In that case you'll need to define what makes your output and reference documents similar. This will likely
come down to the user's intent. Are they asking you for a fact? Then your response better be factual. There is [research into factuality (e.g., paper from a friend of mine)](https://arxiv.org/abs/2104.13346) but it goes beyond the scope of this blog post
nor is it my expertise.


The point is this, if you want to output something besides the text itself, make sure you have a way of verifying it is what the end user wants. In my opinion there is rarely 
a reason to do this as it adds a lot of complexity and will likely require training and evaluating your own verification method. We will cover how to do this in practice in future
blog posts.

### Data Molding and Deduplication

Data molding is the general process of selecting or filtering data. A specific type of molding is deduplication. Deduplication, in the context of two text documents, will refer to the process of identifying and removing two similar texts or documents.
We will discuss these tasks together and how similarity can be used to do them. Below are 4 applications I've encountered that fall into these categories

#### Dataset Deduplication

One use of deduplication is to curate training datasets. This can be used to removed contradictorly labelled data or remove very common data 
from your corpus. In the context of pretraining LLMs, it has been shown to improve performance.

However other papers debate this. 

AFAICT, there hasn't been any extensive study as to why deduplication can improve performance and what deduplication techniques work best. Typically minhash
is used and settings are copied from one paper to another. Due to the limited amount of information available and not having conducted
rigorous experiments myself, I can't really apply the similarity framework here. If you can afford it, I would recommend experimenting
with different deduplication methods and training one or more models to see the effect. How realistic this is will likely depend on the task
and amount of data.

#### Source of Record Deduplication

Deduplication might also be used for system of records. By system of record I mean any kind of wiki, database, or collections of documents. One example of a collection of documents you'll be familiar
with are laws. In my experience the goal of this type of cleaning
1. Make searching the system easier (less to transmit/parse)
2. Avoiding any confusion. For example, if two laws are discussing the same subject what happens if they contain contradictions

Unlike dataset deduplication, where improved downstream performance is the goal, here the priority the end users experience.
Moreover, the cost of incorrectly removing data is high so we may want to be rather strict about what is considered a match. Some records may look alike, but be distinct for business reasons
not readily apparent in the text. This use case has, in my experience, always has been and should be carefully monitored by experts.

For these reasons I've avoided using meaning based similarity and stuck to syntactic similarity. While it could miss documents
that are conceptually similar, but differently worded, it does a rather good job finding similar documents. There are only so many
ways to discuss a specific subject. By setting thresholds and having experts comb the results, you tend to get a very good system.


#### Reference Based Filtering

Sometimes a corpus or system output must not include a specific example. This is likely how OpenAI is addressing the risk of copyrighted material leaks.

In the context you'll need to define what similar is, whether it's a percentage of text, overall meaning, or syntax.


#### Reference Based Selection 

Sampling datasets for domain transfer

### Plagiarism

This is a subject I'm not familiar with, but none the less find relevant when discussing similarity. I will refer heavily on the work of [Folytnek et al.](https://dl.acm.org/doi/10.1145/3345317) and suggest you take this section 
with a grain of salt. It is based on my understand of textual similarity rather than experience with plagiarism detection. 

Folytnek et al. set out five types of plagariasm. I've taken the liberty of tagging them as one of the three types of similarity. For specific methods I would refer you to the survey cited and recent literature. The idea of this mapping is just to highlight the types of tools appropriate for specific types of plagiarism and how to best detect them.


1. Characters-preserving plagiarism (Syntactic)
• Literal plagiarism (copy and paste)
• Possibly with mentioning the source
2. Syntax-preserving plagiarism (Syntactic)
• Technical disguise
• Synonym substitution
3. Semantics-preserving plagiarism (Meaning)
• Translation
• Paraphrase (mosaic, clause quilts)
4. Idea-preserving plagiarism (Meta-textual)
• Structural plagiarism
• Using concepts and ideas only
5. Ghostwriting (Meta-textual)


### Information Retrieval

This subject is one of the first examples that comes to mind when discussing similarity methods. Information retrieval, in it's simplest and most common form, consists
of taking a query and returning the most relevant documents. Depending on the query, this could require any of two types of textual information:
1. Syntactically: Often the words we choose to query will show up in a document, so just finding documents that contain those words does pretty well
2. Meaning: Synonymous words or concepts won't be captured by syntax based approaches. "Cutest cat breed" will not match results that only contain the words "kitten", which could still contain relevant information.

I can't really think of a context where a query would contain Meta-textual information, but I may just lack imagination. Feel free to comment below if you think of one.

#### Multilingual IR

[Here's a summary of Multilingual IR (MLIR) courtesy of Bob Frederking](https://www.cs.cmu.edu/~ref/mlim/chapter2.html). While the difference between IR and MLIR may seem  minor it has interesting implications.
The two texts retrieved will not only not be written in the same way,
they will not be in the same language, maybe not even in the same alphabet or character set.  This poses additional 
challenges not seen by IR. It is however distinct from translation, in that you are not necessarily just trying to find an equivalent text, but rather meaningful results WRT to that text.
Meta-textual elements might come into play here as well, because idioms or common expressions need to be translated in a meaningful way. Different audiences may also have different
frames of reference (popular shows, etc..) which could shape which questions are asked in a given language (or region).

#### Question Answering

WIP, still thinking about whether using similarity for QA is a separate task or something fundamentally related to IR.

### Translation

Translation is not a task I have a lot of experience with, especially framing it as a retrieval problem. The contexts in which I've seen translation framed as retrieval are: 
1) Retrieving the nearest match in a pool of candidates. While this may seem odd, as it does limit the output space, this can make sense from a safety/liability point of view. No need to rely on an online verification process if you have a finite universe of acceptable outputs
2) [Creation of datasets through the alignment of monolingual corpora](https://arxiv.org/abs/1711.00043)
3) [As part of a translation system, either to generate candidates or give additional information to the model](https://aclanthology.org/N18-1120/)

Translation I think involves meaning and the Meta-textual. Meaning as you need to capture the 5Ws and meta-textual for everything related to cultural references. If you fail to convery the important facts,
that defeats the point of communication in general. As for the Meta-textual, I think it's important when you are trying to be understood, idioms are one example of this. Another one, if someone made a reference to a German 60s pop band
as an example of a popular band, it would likely be lost on anyone who isn't a German speaker or familiar with German culture from the 60s. It might make sense to rank a document mentioning a reference familiar to the audience (linguistically,
culturally, age-wise, etc..).



### Mapping {#mapping}

Mapping is again a task I haven't come across in literature per say, but it would likely fall under the umbrella of "semantic similarity". Mapping in this context
is to finding if two different types of documents are related to one another.

At work I often need to map regulatory texts to compliance texts (essentially a procedure that makes sure the company is following the law). The idea is to make sure an organization has covered all the appropriate laws with procedures.

We couldn't frame this problem as syntactic or meaning based similarity if we tried.
1. These two texts don't look a like, they don't use the same wording
2. They don't even discuss the same things. The regulation describes what must not happen and the procedure describes what is being done without ever referencing what it prevents explicitly


A more day-to-day example of why mapping things together is hard is that I can describe a trip to the grocery store without ever using the words grocery store. You could probably avoid any wording that would make it clear you are shopping and yet still have someone understand it from context. This is hard and requires world knowledge
I have not seen any model be able to bridge this gap. Often the solution involves hand coding or including metadata and using algorithms to constrain on what pairs to compare.


### Clustering

Clustering goes beyond the text <> text setting, we've discussed, but I mention this downstream use of similarity for completeness. It often comes up when discussing
similarity and unsupervised learning seems to be in vogue at the moment.

Clustering can be done on any level (syntactic, semantic, or Meta-textual). The method you use to represent text or find simialrties between texts will determine which of these
(it's unclear if/how well they can be decoupled).

To be frank, I'm skeptical of clustering, finding structure in any kind of data, forget text, is difficult. 

First multiple structures can exist. There is a NYT game called "Connections", where 16 words are presented.
You group together words into 4 sets of 4. There are often multiple set of 4 that could be valid or alternate labels which aren't. Now imagine how complicated this becomes
when we are talking about a whole document rather than a single word.

Second the issue is finding meaningful patterns. Your model will find patterns, most of which (if not all) in my experience are meaningless. 
I've tried clustering regulations from different regulators using a variety of embeddings/techniques and all that we built was a very powerful way of grouping
together data from the same regulator or vaguely related concepts. It could not find a taxonomy of regulations like a human would.

While there are ways of trying to distill meaning from the patterns found, this is often a one off exercise. You will need to either convert this into a hierarchy
and train a model to appropriately classify text or repeat this exercise every time you get new data. In practice if you can define what you want to group together, you should save yourself time and label data.


## Conclusion

We've discussed and defined 3 broad categories of textual similarity and shown some example tasks and how that framework can be applied.

Next step is to discuss how to translate these characteristics into algorithms and implement similarity in practice. The post will discuss
algorithms and a way of framing them together.