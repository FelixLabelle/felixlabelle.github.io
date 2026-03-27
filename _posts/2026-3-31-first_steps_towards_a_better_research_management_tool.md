# WIP: Steps Towards  Better Personal Information Access for Papers

Conducting surveys or finding research papers to answer a given research question can be a pain in my experience.
There are concretely four steps:
0. Defining the scope
1. Finding relevant papers within that scope
2. Triaging, collating, and organizing relevant papers
3. Interpreting results; either finding a conclusion, refining scope (return to 0.) or identifying further research needed.

While each step is difficult, 1 and 2 are unnecessarily hard given the current technology.

## Problem Definition

Currently searching for papers is frustating (to me atleast). I need to use different tools and organize the results 
manually. I've tried different combinations of search methods, namely
1. Google
2. Arxiv
3. Semantic Scholar
4. Alphaxiv
5. Connected papers
6. Deep research tools (e.g., deep research, etc..) <!-- TODO: Add more examples -->;

[Aaron Tay gives some insight
into the opaqueness of current search approaches that IMO is partially what makes surveying fields harder](https://aarontay.substack.com/p/the-blank-box-problem-why-its-harder).
In short, under the hood each of these tools uses different technologies such as keyword search (e.g., boolean search, bm25), neural embeddings, citation based methods, and now agents. Each expects different input formats and is lacking in slightly different ways. In my experience that translates into needing a large number of queries and tools is needed to perform an exhaustive search. Due to the differences in capabilities and the strengths of each not being well communicated
it causes issues. 

Here's kind of a short list of the weaknesses of these different approaches in my experience.
1. Keyword based methods are pretty intuitive and give great results... as long as you have the correct keywords...
2. Neural methods can be used to perform QA or other tasks. In general they have a fair amount of flexibility. That said they are often trained to perform a specific task 
on specific data and are likely to fail in unexpected ways outside of their domain. My favorite failure mode so far has been ranking punctuation heavy spans very highly.
3. Bibliographic methods are cool and leverage scientific communities. That said they tend to not be interdisciplinary and can wind up only finding certain pools of authors
that co-cite each others work. If there are cliques, veins of research, or different communities you may not find them
4. Deep research/agentic approaches are relatively new. Very similar to multihop QA IMO. Still think this is a naiscent area. I've been really dissapointed by these tools,
they tend to pull in only tangentially relevant results and miss some clear matches. I think agents may help, but honestly I think having a machine
do the thinking defeats the purpose of a literature survey. The amount of papers proposing agentic researchers seems to indicate I may be old fashioned in thinking this way.

Each method requires an understanding of how the process is conducted and the potential short comings of each. 
In pratice I need to use these tools together. This means different interfaces, tools, and then needing to 
manually combine results. Then comes the other fun part, choosing your system to organize.
Some examples include
1. An excel sheet (I still use this, it's flexible, simple, shareable, and pretty universal)
2. zettlekasten
3. zotero 
4. semantic scholar (I like the idea of this tool more than the tool itself)

tl:dr; there are two distinct promblems; selection and curation. Below I outline a first attempt at smoothing out the search experience.

## Custom Solution

I think it's easy to fall into the trap of a better 
method being all that's needed, but I don't think that's the case here.
Rather than try to create a perfect search method, the approach I chose to tackle the approach is 
to offer more search options and methods in one spot (site).

Alright, so this requires building a search engine and the underlying components required 
to maintain one. At a high-level, there are three parts to the pocess
1. Corpus ingestion & curation
2. Indexing
3. User experience

### Corpus ingestion & curation

#### Crawling

Ok, I kind of rushed this part, I just wanted a PoC at first. I decided to use Arxiv only.
 Yes I'm aware I could use Semantic scholar. 
I chose not to do this in case for simplicity and the flexbility to build out my
own pipeline in the future.

I may go back on this, since the goal is eventually to write a paper and using arxiv 
only limits the corpus (IMO this is relatively limited since most people write preprints) but more importantly requires extra work 
to generate citations. I've also been had by AI2 tools being shuttered before so I might not (RIP AllenNLP). TBD.

The method to crawl and download arxiv preprints is pretty straightforward.
1. Download a list of arxiv papers from [kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Run over the entire list
3. Import all previously unseen (i.e., new) arxiv ids and versions into a database, specifically a raw storage table

#### Storage

To store results you need a storage method. I opted for a single database to warehouse both raw data 
and processed data. The structure is pretty straightforward, here's a relational diagram show casing it:
<!-- TODO: Add diagram -->

To power this I used postgresql, specifically paradedb. Honestly there are too many products for vector search.
My criteria was pretty straightforward
* SQL database since I wanted to store app data in a relational way as well
* Support multiple indices and types of indices, namely dense vectorial, sparse vectorial, BM25
* Works


It's been working well. My only complaint is that queries can take 15 seconds+ sometimes, but thismay well be the hardware I'm running the DB on
(a 2 core dell wyse 5070). I'm in the process of migrating.

#### Filtering

Ok so the data has been stored, but we still need to choose relevant results. 
Some optimisic (naive) napkin math based on a sample led me to believe I could store the entire dataset..
After downloading more PDFs, I quickly realized that was not the case.

So I developped a heuristic for choosing with arxiv PDFs to download. Specifically tags are used.
<!-- TODO: INclude code for choosing tags -->

### Indexing

Earlier I buried the lead a bit. There aren't just 5 methods of searching, there are hundreds.
Beyond the high-level categories I outlined above, each broad category contains methods and other variants.
There is also a question of how and when they are applied. Below is a table

These variations 
data they are run over.

| End Use Case   | Method Family | Technique / Model        | Input Field(s)            | Representation Type       | Index Type              | Query Type             | Scoring Available | Scoring Type                          | Fusion / Rerank Stage | Notes |
|----------------|--------------|--------------------------|---------------------------|----------------------------|--------------------------|------------------------|-------------------|----------------------------------------|------------------------|-------|
| Search         | Keyword      | BM25                     | title, abstract, body     | Sparse (token-based)       | Inverted Index           | Keyword                | Yes               | Explicit (BM25: TF-IDF variant)        | Optional               | Strong lexical baseline |
| Search         | Keyword      | PostgreSQL FTS           | title, body               | Sparse                     | Inverted Index           | Keyword                | Yes               | Explicit (TF-IDF variant)              | Optional               | Easy DB integration |
| Search         | Vector       | SPLADE                   | abstract, body            | Sparse (learned expansion) | Inverted Index           | Text → sparse vector   | Yes               | Explicit (dot product)                 | Optional               | Learned term expansion |
| Search         | Vector       | SPLADE (field-specific)  | title                     | Sparse (learned expansion) | Inverted Index           | Text → sparse vector   | Yes               | Explicit (dot product)                 | Optional               | Field-tuned retrieval |
| Search         | Vector       | Nomic embeddings         | abstract, body            | Dense                      | ANN (HNSW / IVF)         | Text → dense vector    | Yes               | Explicit (cosine / dot product)        | Optional               | Semantic retrieval |
| Search         | Exact        | SQL equality / filters   | structured fields         | Exact / structured         | B-tree / Hash            | Structured query       | Yes               | Explicit (boolean / exact match)       | Optional               | Deterministic filtering |
| Recommendation | Citation     | Bibliographic Coupling   | citation graph            | Graph-based                | Graph index              | Node similarity        | Yes               | Explicit (overlap, normalized counts)  | Often                  | Forward-looking similarity |
| Recommendation | Citation     | Co-citation              | citation graph            | Graph-based                | Graph index              | Node similarity        | Yes               | Explicit (co-occurrence counts)        | Often                  | Backward-looking similarity |
| Recommendation | Hybrid       | Fused (normalized)       | multi-field / multi-index | Mixed (sparse + dense)     | Multi-index              | Multi-query            | Yes               | Explicit (weighted / normalized score) | Yes                    | Combines retrieval signals |
| Summarization  | Generative   | LLM summarization        | full document             | Dense (internal latent)    | None                     | Prompt                 | No                | None (generative output)               | Optional               | Produces text, not ranking |
| Clustering     | Vector       | KMeans (embeddings)      | abstract, body            | Dense                      | None (batch)             | N/A                    | No (implicit)     | Implicit (distance to centroid)        | Optional               | Offline grouping |
| Clustering     | Graph        | Community detection      | citation graph            | Graph-based                | Graph index              | N/A                    | No (implicit)     | Implicit (modularity optimization)     | Optional               | Structure-driven clusters |


The number of combinations becomes a bit absurd when you start actually
thinking about it. That's before we even start talking about combining methods through fusion
or reranking. no agents yet. just wanted to try a better search

### UI/UX

The frontend is simpistic, both visually and in terms of structure. There are 3 high-level screens

#### Search

#### Library

Ability to create recommendations

Recommendation engines:
1. vector based
2. bibliometric

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
and that instead the performance comes slowly from the fact that a much less lossy, but much larger representation is used to do search.

Query rewriting. dOc augmentation are also possibilities.

### DSL for Search

One of the things that's hard about search is that it isn't one single task,
it's a large myriad of tasks masquerading as one. A given query from different users
could have different valid results, different inputs could map to different
Intent is hard to deciper from a query.

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
