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
In practice I need to use these tools together. This means different interfaces, tools, and then needing to 
manually combine results. Then comes the other fun part, choosing your system to organize.
Some examples include
1. An excel sheet (I still use this, it's flexible, simple, shareable, and pretty universal)
2. zettlekasten
3. zotero 
4. semantic scholar (I like the idea of this tool more than the tool itself)

tl:dr; there are two distinct problems I want to tackle; selection and curation.

## Custom Solution

I think it's easy to fall into the trap of a better 
method being all that's needed, but I don't think that's the case here. I'm not sure approach, outlined 
above, can return all relevant results.
Rather than try to create a perfect search method, the approach I chose to tackle the approach is 
to offer more search options and methods in one spot (i.e., site) and see what works and doesn't in the long term.
To be clear, I don't think this search experience is for everyone. There are a lot of choices that can be made and honestly
they will probably make the process more confusing (even for me having more than 2-3 choices is a lot).

This translates into building a search engine and the components required 
to maintain one. At a high-level, there are three parts to the process
1. Corpus ingestion & curation
2. Indexing
3. User experience

### Corpus ingestion & curation

Ingestion and curation sounds misleadingly simple. Just taking a look at the 
ingestion pipeline below will give you an idea of the scale of work required.

<details markdown="1">
<summary>Update Script</summary>

```
# Crawling
kaggle datasets download Cornell-University/arxiv --unzip --force

# Data intake and preprocessing
python insert_raw_data.py
python score_relevance.py
python download_pdfs.py
python extract_pdfs.py


# Data transfer
python insert_docs.py

# Add metadata only to search data
python add_semantic_scholar_metadata.py -t SEARCH_CORPUS_TABLE_DATA


# Span creation
python create_spans.py -t SEARCH_CORPUS_TABLE_DATA -f text -s regex -n 512 -o 0
python create_spans.py -t SEARCH_CORPUS_TABLE_DATA -f text -s regex -n 512 -o 256
python create_spans.py -t SEARCH_CORPUS_TABLE_DATA -f text -s regex -n 256 -o 0
python create_spans.py -t SEARCH_CORPUS_TABLE_DATA -f text -s regex -n 256 -o 128

# Indexing
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_0_DATA -f span_text -m "text-embedding-nomic-embed-text-v1.5"
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_0_DATA -f span_text -m "BM-25" --force
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_128_DATA -f span_text -m "text-embedding-nomic-embed-text-v1.5"
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_128_DATA -f span_text -m "BM-25" --force
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_0_DATA -f span_text -m "text-embedding-nomic-embed-text-v1.5"
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_0_DATA -f span_text -m "BM-25" --force
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_256_DATA -f span_text -m "text-embedding-nomic-embed-text-v1.5"
python embed_db.py -t SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_256_DATA -f span_text -m "BM-25" --force
python embed_db.py -t SEARCH_CORPUS_TABLE_DATA -f text abstract -m "BM-25" --force
python embed_db.py -t SEARCH_CORPUS_TABLE_DATA -f abstract -m "text-embedding-nomic-embed-text-v1.5"
python embed_db.py -t SEARCH_CORPUS_TABLE_DATA -f abstract -m "naver/splade-cocondenser-ensembledistil"

# Create graph for bibliometric recommender
python create_citation_graph.py -t SEARCH_CORPUS_TABLE_DATA --force

# Dsiplay results
python analyze_pipeline.py
```
</details>

To make sure the dataset stays up to date the code is designed to store state in the database and reprocess items 
that have failed or new items. This means that after the first run, the number of files run is minimal. I use 
cron to run the update job on Saturday evenings.

<details markdown="1">
<summary>Cron Job for Running Update Pipeline</summary>
```
0 2 * * 0 source ~/arxiv/arxiv_env/bin/activate && /home/felix/research_assistant/src/intake_pipeline/update_pipeline.sh
```
</details>

#### Crawling

Ok, I kind of rushed this part, I just wanted a PoC. I decided to use papers off of arxiv only.
 Yes I'm aware I could use Semantic scholar. 
I chose not to do this in case for simplicity and the flexbility to build out my
own pipeline in the future. I've also been had by abandoned AI2 tools before (RIP AllenNLP).

I may go back on this, since the goal is eventually to write a paper and using arxiv 
only limits the corpus IMO this is issue is relatively limited since most people write preprints; more importantly requires extra work 
to generate citations. 

The method to crawl and download arxiv preprints is pretty straightforward.
1. Download a list of arxiv papers from [kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Run over the entire list
3. Import all previously unseen (i.e., new) arxiv ids and versions into a database, specifically a raw storage table

#### Storage

For storage I opted for a single database to warehouse both raw data 
and processed data. The structure is pretty straightforward, here's a UML diagram show casing it:

<details markdown="1">
<summary>UML Relational Diagram</summary>
<pre class="mermaid">
erDiagram
    %% Generated from live public schema foreign keys
    %% Unenforced logical links from db_schemas.py are not included here
    %% M2M_QUERY_ARXIV_PAPER = _m2m_query_arxiv_paper

    M2M_QUERY_ARXIV_PAPER {
        text mapping_id PK
        text query_id FK
    }

    BIBLIOMETRY_TABLE {
        text pair_id PK
        text p1_arxiv_id FK
        text p2_arxiv_id FK
    }

    DOCUMENTS {
        text arxiv_id PK
    }

    DOMAIN_WORKFLOW_DATA {
        text arxiv_id PK
    }

    LIBRARY {
        text entry_id PK
    }

    PIPELINE_VERSIONS {
        integer id PK
    }

    QUERIES {
        bigint query_id PK
    }

    QUERY_RESULTS {
        bigint result_id PK
        bigint query_id FK
        bigint span_id FK
        text doc_id FK
    }

    QUERY_TABLE {
        text query_id PK
    }

    RAW_DATA {
        text arxiv_id PK
    }

    RESEARCH_QUESTIONS_DATA {
        text rq_id PK
        text arxiv_id FK
    }

    RESEARCH_TOPIC_MAPPING {
        text mapping_id PK
        text topic_id FK
    }

    RESEARCH_TOPICS {
        text topic_id PK
    }

    SEARCH_CORPUS {
        text arxiv_id PK
    }

    SPANS {
        bigint span_id PK
        text arxiv_id FK
    }

    SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_0 {
        text span_id PK
        text arxiv_id FK
    }

    SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_128 {
        text span_id PK
        text arxiv_id FK
    }

    SPANS_SEARCH_CORPUS_TEXT_REGEX_512 {
        text span_id PK
    }

    SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_0 {
        text span_id PK
        text arxiv_id FK
    }

    SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_256 {
        text span_id PK
        text arxiv_id FK
    }

    SPANS_SEARCH_CORPUS_TEXT_WHITESPACE_512 {
        text span_id PK
    }

    QUERY_TABLE ||--o{ M2M_QUERY_ARXIV_PAPER : "query_id to query_id"
    RAW_DATA ||--o{ BIBLIOMETRY_TABLE : "p1_arxiv_id to arxiv_id"
    RAW_DATA ||--o{ BIBLIOMETRY_TABLE : "p2_arxiv_id to arxiv_id"
    DOCUMENTS ||--o{ QUERY_RESULTS : "doc_id to arxiv_id"
    QUERIES ||--o{ QUERY_RESULTS : "query_id to query_id"
    SPANS ||--o{ QUERY_RESULTS : "span_id to span_id"
    SEARCH_CORPUS ||--o{ RESEARCH_QUESTIONS_DATA : "arxiv_id to arxiv_id"
    RESEARCH_TOPICS ||--o{ RESEARCH_TOPIC_MAPPING : "topic_id to topic_id"
    DOCUMENTS ||--o{ SPANS : "arxiv_id to arxiv_id"
    SEARCH_CORPUS ||--o{ SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_0 : "arxiv_id to arxiv_id"
    SEARCH_CORPUS ||--o{ SPANS_SEARCH_CORPUS_TEXT_REGEX_256_OFFSET_128 : "arxiv_id to arxiv_id"
    SEARCH_CORPUS ||--o{ SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_0 : "arxiv_id to arxiv_id"
    SEARCH_CORPUS ||--o{ SPANS_SEARCH_CORPUS_TEXT_REGEX_512_OFFSET_256 : "arxiv_id to arxiv_id"

</pre>
</details>

Honestly there are too many products for vectorial search; my criteria was the following:
* SQL database since I wanted to store app data in a relational way as well
* Support multiple indices and types of indices, namely dense vectorial, sparse vectorial, BM25
* Straightforward to use (i.e., not use multiple programs and need to sync them up)
* Open source


That cuts down the list of options quite a bit and paradedb (or pgvector) was clear winner for me.
I had read some complaints about performance, but this may well be old news.
[Performance seems comparable to Elastic search (in certain respects according to this benchmark)](https://github.com/inevolin/ParadeDB-vs-ElasticSearch) 
and it has been working well. My only complaint is that queries can take 15 seconds+ sometimes, but this may well be the hardware I'm running the DB on
(a 2 core dell wyse 5070) and the fact I haven't indexed the vector columns.

#### Filtering

Once the raw data has been put stored, we still need to choose relevant results for downloading
and further processing. Some optimisic (naive) napkin math based on a sample led me to believe I could store all of arxiv.
After downloading a hundred thousand PDFs, I realized that was not the case.
A heuristic for choosing with arxiv PDFs to download was in turn used: 
<!-- TODO: INclude code for choosing tags -->

<details markdown="1">
<summary>Relevance Scoring Mechanism</summary>
```
relevant_categories = set(["cs.CL", "cs.FL", "cs.HC", "cs.IR", "cs.LG"])


def score_relevance_v1(items):
    relevance_items = [
        {
            "arxiv_id": item["arxiv_id"],
            "relevance": all(
                [cat in relevant_categories for cat in item["categories"]]
            ),
            "_relevance_score": float(
                all([cat in relevant_categories for cat in item["categories"]])
            ),
            "_relevance_confusion": 0.0,
            "_relevance_model": "heuristic_v2",
        }
        for item in items
    ]
    return relevance_items

```
</details>

This leaves a corpus of ~100K documents. There are still irrelevant documents IMO and some results missed,
but for now the size required (~300GB for the PDFs and ~100GB for the DB) is potable.

#### Downloading and Preprocessing

PDFs are downloaded.

Text is extracted (use pymupdf).

Papers that have text extracted are migrated to the search corpus

Additional metadata for citations and references is added.

Spans of the documents are created. Originally I wanted all spans to be in one table. I noticed that query speed is directly 
related to the number of items. Even with an index, larger number of items are heavily penalized. Rather than 

### Indexing


Ok so, the data has been processed and downloaded. Next step is to make it searchable.

Earlier I buried the lead a bit. There aren't just 5 methods of searching, there are hundreds.
Beyond the high-level categories I outlined above, each broad category contains methods and other variants.
Methods can be applied to the title, abstract, body, spans, etc..
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

The number of combinations becomes a bit absurd when you start actually
thinking about it. That's before we even start talking about combining methods through fusion
or reranking. no agents yet. just wanted to try a better search

Bibliographic coupling was also used for "indexing".

### UI/UX

The frontend is simplistic, both visually and in terms of structure.
The first sight is the landing page that provides 3 additional screens.

![Research hub landing page](/images/research_hub_homepage_site.png)

#### Search

The search page
![Research hub landing page](/images/research_management_tool_search_papers_site_empty.png)

Ability to select a large number of index + source combinations

![Research hub landing page](/images/search_papers_results_site_dropdown_open.png)

#### Library


Library is still pretty rudimentary, don't even have a recommendation there yet.

![Research hub landing page](/images/search_papers_results_site_dropdown_open.png)

Ability to add, remove papers. Can also assign them to RQs. Currently using it to triage 
papers until they find a more permanent home elsewhere.

#### Research Question Management

RQ landign page has RQs
![Research hub landing page](/images/research_questions_list_site_active.png)

RQ has papers 

![Research hub landing page](/images/research_question_attached_papers_site_empty.png)

Ability to create recommendations

![Research hub landing page](/images/)research_question_sheets_site_recommendations_results_5.png
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

### Data Analysis and Data Gathering

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

Beyond the methods themselves, understanding how to interact with each approach
or when to use which is an issue. I have ideas for both.

Next word prediction that's corpus and method aware. E.G., using a logit biased method to help suggest
relevant keywords. Or using a predictive model that's trained on the same corpus as the embedding.
This might go beyond 

I think an agent might be the best way of simplifying the approach, have a model generate the abastraction for users.
One part of that might be abstracting search.


### DSL for Search

One of the things that's hard about search is that it isn't one single task,
it's a large myriad of tasks masquerading as one like similarity. A given query from different users
could have different valid results, different inputs could map to different
User intent is hard to decipher from just a query. Moreover not all searches 
are as simple as just finding results, [like tip of the tongue search](https://arxiv.org/abs/2502.17776).
<!-- TODO: List other examples of tasks -->

Evidence of this
https://arxiv.org/pdf/2602.00296
https://arxiv.org/abs/2506.13743

I think using the right tool to the solve the task goes beyond just having the best embedding
or search method. Intent needs to be understood and the users intent needs to be potentially clarified through direct interaction. I think agents 
(or a similar system) may be more capable of this.

Rather than use an agent (or other system) to directly to manipulate results, it might make more sense to use them with a better
tool that gives flexibility on the right tool(s) to use. You're not going to do QA, simple keyword retrieval with the same tool.
Sure an LLM might be able to help bridge the game, but that creates a failure point.  What if the system has access to a higher-level
search langauge to pick the right tool for the job. 

While systems do exist to pick intent, as far as I can tell there isn't one with a large level of flexibility.
Current LLMs can write code (to a certain extent), so why not have code to create search pipelines?
A DSL (domain specific language) for search and recommendations would be a good opportunity for this.
DSL gives simple and more complex systems flexibility.

A DSL can also be used as part of an RL pipeline (I think, lol RL is not my specialty).

### More Search Tools

The approach used so far is rather simplistic and there exist a large number of other approaches.
Here are the ones I'd like to implement:
1. Cross encoder
2. Listwise prompt based reranker
3. Pair wise prompt based reranker
4. Colbert
5. Custom dense embeddings that allow for different question formats 
6. Query rewriting/augmentation
7. Document augmentation

Some of these are more complex then others. ColBERT requires additional indexing not
setup in paradedb. Most other methods can be supported with my current stack,
they just require additional code.

### Beyond Search

Rather than search we can use workflows to do systematic surveys over the entire corpus.
The approach is simple
1. define criteria for relevant paper that can be translated into tags/classes
2. design a workflow that uses simple conditional logic to filter results
3. implement pipeline
	1. implement skeleton for pipeline
	2. create initial classification prompts
	3. run pipeline
	4. review each classification stage to make sure that results are adequate
4. review the pipeline

This approach allows you to set a wider net, estimate global performance (risk of non-inclusion)
using traditional metrics we are all familiar with. This likely isn't a new approach, but 
I think the trick here is how to make this approach work reliably and quickly. I'm also trying to use this 
approach for my domain survey. 

## Closing Remarks

I plan on sharing the underlying code when it's further along. Right now I'm just trying to get a tool that works 
well enough for me and allows experimentation with my personal preferences/needs for search. I'm currently using it to survey definitions of domains in NLP, so 
there will likely be more updates after I get done with that.