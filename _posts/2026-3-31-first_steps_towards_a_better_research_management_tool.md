# WIP: Steps Towards  Better Personal Information Access for Papers

Conducting surveys or finding research papers to answer a given research question can be a bit of a pain in my experience.
There are four steps:
0. Defining the scope
1. Finding relevant papers within that scope
2. Triaging, collating, and organizing relevant papers
3. Interpreting results; either finding a conclusion, refining scope (return to 0.) or identifying further research needed.

While each step is difficult, 1 and 2 are unnecessarily hard given the current technology.
When conducting a survey are a number of different tools and underlying methods to find relevant papers.
 Common methods include keyword search (e.g., boolean search, bm25), neural embeddings, citation based methods, and now agents.
 Each method requires an understanding of how the process is conducted and the potential short comings of each. 

## Problem Definition

Currently searching for papers is frustating (to me atleast). I need to use different tools and organize the results 
manually. I've tried different combinations of search methods, namely
1. Google
2. Arxiv
3. Semantic Scholar
4. Alphaxiv
5. Connected papers
6. Deep research tools (I don't use actively)

I find each is lacking in slightly different ways. [Aaron tay gives some insight
into why that is in this blog](https://aarontay.substack.com/p/the-blank-box-problem-why-its-harder).
Due to the differences in capabilities and the strengths of each not being well communicated
it causes issues.

I need to use these tools together. This means different interfaces, tools, and then needing to 
manually combine results. Then comes the other fun part, choosing your system to organize.
Some examples include
1. An excel sheet (I still use this, it's flexible, simple, shareable, and pretty universal)
2. zettlekasten
3. zotero 
4. semantic scholar (honestly this tool is probably the closest to my personal preference)

These two problems are relatively simple to solve for IMO.

## Custom Solution

Fundamentally there each have strengths and weaknesses asfaict.
Here's kind of a short list of these
1. Keyword based methods are pretty intuitive and give great results, as long as you have the correct keywords
2. Neural methods can be used to perform QA or other tasks. In general they have a fair amount of flexibility. That said they are rather brittle
There is also a fair amount of flexibility within these systems 
3. Bibliographic methods are cool and leverage scientific communities. That said they tend to not be interdisciplinary and can wind up siloed
4. Deep research/agentic approaches are relatively new. Very similar to multihop QA IMO. Still think this is a naiscent area. I've been really dissapointed by these tools,
they tend to pull in only tangentially relevant results and miss some clear matches. I think agents may help, but honestly I think having a machine
do the thinking defeats the purpose of a literature survey.

I think it's easy to fall into the trap of a better 
method being all that's needed, but I don't think that's ever the case.
Rather than try to create a perfect search method, the approach I chose to tackle the approach is 
to offer more search options and methods in one site.

Alright, so this requires building a search engine and the underlying components required 
to maintain one. At a high-level, there are three steps
1. Corpus ingestion & curation
2. Indexing
3. User experience

### Corpus ingestion & curation

#### Crawling

Arxiv only. Semantic scholar is an option, but honestly I chose to not opt for this.
I want more control over the stack.I think this has some downsides (e.g., when it comes to citations,

#### Storage

Use paradedb. Wanted to support multiple indices. Wanted relational db 
for app. Honestly it's been working fine. Speed is a bit slow, but that 
may well be the hardware I'm running on.

#### Filtering

Tag for relevance, using a heuristic based on tags.
Tried a finetuned approach, but it was hard to get good labels.
Will try a synthetic dataset.

### Indexing

Different combos of text and indexing 

There are a handful of indices I used:
* splade
* BM25
Sparse

BM25

Splade

Dense

nomic

custom embedding

BERT

Recommendation


### UI/UX

#### Search

#### Library

Ability to create recommendations

Recommendation engines:
1. vector based
2. bibliobased

#### Research Question Management

Ability to create recommendations

Recommendation engines:
1. vector based
2. bibliobased

## Next Steps

This was a good first thrust, but there are still a number of issues that need to be addressed.

### Large Scale Usage

I have 3 RQs I'm currently using this for. It's worked pretty well so far and surface many papers 
I haven't found through my other typical tools. That being said, 3 RQs is not a lot and even the largest
only has 20 papers. I'm currently planning a large scale survey for domain which I expect to net 100+
papers.

The current scale of the app makes it easy to neglect many of the projects.

### Data Quality

I used a pretty naive approach to create my dataset and monitor its health. The biggest short comings I see are the following
1. A single source was used, yeah some papers are going to be missing
2. The ingestion pipeline 
	1. Only part of arxiv is used, currently tags are used but very naive heuristic. Train a model for relevance
	2. The pipeline fails. This is tracked, but currently my system makes it hard to fix individual failures
	3. Pull in citations and some other information from outside sources, it looks like some data may be missing 
3. There are no individual or corpus level quality checks
	1. At the individual level
		1. No checks for methodology
		2. Some preprints are very wonky, there is everything from class projects to rants
	2. Corpus level
		1. Dedupe

### UI/UX

The UI/UX is the next step 

### More Search Tools

Reranking, namely co-embedders.

Colbert. This will likely take much longer since it would require a different .

Better dense embeddings. Having just mentionned ColBERT this may seem odd. I'm personally not as ColBERT pilled as NLP twitter.
I  believe that ColBERT is better in specific conditions, but that for a well defined task dense embeddings likely 
have similar performance at much lower cost. I don't personally believe there is anything special about late interaction 
and that instead the performance comes slowly from the fact that a much less lossy representation is used to do search.

### DSL for Search

One of the things that's hard about search is that it isn't one single task,
it's a large myriad of tasks masquerading as one. A given query from different users
could have different valid results, different inputs could map to different

I think using the right tool to the solve the task goes beyond just having the best embedding
or search method. LLMs have started filling this role, but I think it's a bit short sighted
solution. Often it;s unclear to know if the either corpus was searched, if results are reliable, etc..

Currently need to manual select methods.

Rather than use a model directly to manipulate results, it mgiht make more sense to use them with a better
tool that gives flexibility on the right tool(s) to use.

DSL gives simple and more complex systems flexibility.

### Beyond Search

Rather than search, sometimes we can use more system approaches to find relevant papers.
One approach I've been experimenting is workflows to do systematic surveys over the entire corpus.
Using classifiers, creating filter values, filtering based on results, and then using the resulting 
corpus. This has proven to be challenging for a number of reasons.


## Closing Remarks

I might share this code when it's further along. Right now I'm just trying to get a tool that works 
well enough for me. I'm currently using it to survey definitions of domains in NLP, so 
there will likely be more updates after I get done with that.