# WIP: Towards  Better Reserach Management Tools

Conducting surveys or finding research papers to answer a given research question can be a pain in my experience.
This is partly due to the headaches of needing many tools to find all relevant results and the pain of juggling and organizing those results.

IMO there are four steps to a survey:
1. Defining the scope
2. Finding relevant papers within that scope
3. Triaging, collating, and organizing relevant papers
4. Interpreting results; either finding a conclusion, refining scope (return to 0.) or identifying further research needed.

While each step is difficult, 2 and 3 are unnecessarily hard given the current technology.

## The Status Quo

Currently searching for papers is frustrating (to me at least). I need to use different tools and collate the results manually. I've tried different combinations of search methods, namely
1. Google
2. Arxiv
3. Semantic Scholar
4. Alphaxiv
5. Connected papers
6. Deep research tools (e.g., deep research) 

[Aaron Tay gives some insight
into the opaqueness of current search approaches that IMO is partially what makes surveying fields harder](https://aarontay.substack.com/p/the-blank-box-problem-why-its-harder).
In short, under the hood each of these tools uses different technologies such as keyword search (e.g., boolean search, bm25), neural embeddings, citation based methods, and now agents.
Each method expects different input formats and is lacking in slightly different ways.
In my experience that translates into needing a large number of queries and tools is needed to perform an exhaustive search.
Due to the differences in capabilities and the strengths of each not being well communicated


Here's kind of a short list of the failure modes of these different approaches in my experience. 
1. Keyword based methods are pretty intuitive and give great results... as long as you have the correct keywords...
2. Neural methods can be used to perform QA or other tasks. In general they have a fair amount of flexibility. That said they are often trained to perform a specific task 
on specific data and are likely to fail in unexpected ways outside of their domain. My favorite failure mode so far has been ranking punctuation heavy spans very highly.
3. Bibliographic methods are cool and leverage scientific communities. That said they tend to not be interdisciplinary and can wind up only finding certain pools of authors
that co-cite each others work. If there are cliques, veins of research, or different communities you may not find them
4. Deep research/agentic approaches are relatively new. In general I've been really disappointed by these tools,
they tend to pull in only tangentially relevant results and miss some clear matches. I think agents may help, but honestly I think having a machine
do the thinking defeats the purpose of a literature survey.

Each method requires an understanding of how the process is conducted and the potential short comings of each. 
In practice I need to use these tools together. This means different interfaces, tools, and then needing to 
manually combine results. Then comes the other fun part, choosing your system to organize.
Some examples include
1. An Excel sheet (I still use this, it's flexible, simple, shareable, and pretty universal)
2. Zettlekasten
3. Zotero 
4. semantic scholar (I like the idea of this tool more than the tool itself)

Ultimately there are two distinct problems I want to tackle; selection and curation.

## Custom Solution

The next two subsections discuss selection and curation. The selection subsection will discuss
primarily the backend and search methods made available in the tool. The curation section will 
cover the UI/UX around collating data and recommendation engines used to help find additional results.

### Selection

I think it's easy to fall into the trap of a better
method being all that's needed, but I don't think that's the case here.Rather than try to create a perfect search method, the approach I chose to tackle the approach is 
to offer more search options and methods in one spot (i.e., site) and see what works and doesn't in the long term.
To be clear, I don't think this search experience is for everyone. There are a lot of choices that can be made and honestly
they will probably make the process more confusing (even for me having more than 2-3 choices is a lot).

This translates into building a "search engine" and the components required 
to maintain one. At a high-level, there are four parts to making a search tool:
1. Corpus ingestion & curation
3. Indexing
4. User experience

#### Corpus Ingestion & Curation

Ingestion and curation sounds misleadingly simple. 
You just find the results and take them down. As I went along there were always additional steps.
Just taking a look at the 
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

# Display results
python analyze_pipeline.py
```
</details>

To make sure the dataset stays up to date the code is designed to store state in the database and reprocess items 
that have failed or new items. This means that after the first run, the number of documents processed each run is minimal (only new documents). I use 
cron to run the update job weekly (as this is the frequency with which the Arxiv dataset gets updated).

<details markdown="1">
<summary>Cron Job for Running Update Pipeline</summary>
```
0 2 * * 0 source ~/arxiv/arxiv_env/bin/activate && /home/felix/research_assistant/src/intake_pipeline/update_pipeline.sh
```
</details>

##### Crawling

Ok, I kind of rushed this part, I just wanted a PoC. I decided to use papers off of Arxiv only.

Yes, I'm aware I could use Semantic scholar. 
1. I've been had by abandoned AI2 tools before (RIP AllenNLP).
2. I chose not to do this in case for simplicity and the flexbility to build out my
own pipeline in the future. Arxiv was a natural jumping off point

Ok so with the corpus chosen, Arxiv's other advtange is that downloading preprints is pretty straightforward.
1. Download a list of arxiv papers from [kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Run over the entire list
3. Import all previously unseen (i.e., new) arxiv ids and versions into a database, specifically a raw storage table

##### Storage

For storage I opted for a single database to warehouse both raw data 
and processed data. The structure is pretty straightforward, here's a UML diagram showcasing it:

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

Honestly there are too many products for vector search; my criteria was the following:

* SQL database since I wanted to store app data in a relational way as well
* Support multiple indices and types of indices, namely dense vectors, sparse vectors, BM25
* Straightforward to use (i.e., not use multiple programs and need to sync them up)
* Open source


That cuts down the list of options and [paradedb](https://www.paradedb.com/) (or [pgvector](https://github.com/pgvector/pgvector)) was a clear winner for me.


I had read some complaints about performance, but this may well be old news.
[Performance seems comparable to Elastic search (in certain respects according to this benchmark)](https://github.com/inevolin/ParadeDB-vs-ElasticSearch) 
and it has been working well. My only complaint is that queries can take 15 seconds+ sometimes, but this may well be the hardware I'm running the DB on
(a 2 core dell wyse 5070) and the fact I haven't indexed the vector columns.

##### Filtering

Once the raw data has been stored, we still need to choose relevant results for downloading

and further processing. Some optimistic (naive) napkin math based on a sample led me to believe I could store all of arxiv.
After downloading a hundred thousand PDFs, I realized that was not the case.
A heuristic for choosing with Arxiv PDFs to download was in turn used: 

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

##### Downloading and Preprocessing

Once we've identified relevant papers, the next step is to download the PDFs. After downloading 
all relevant PDFs, their text is extracted using [PyMuPDF](https://github.com/pymupdf/PyMuPDF).

Papers that have text extracted are migrated to the search_corpus table. Afterwards additional metadata for citations and references is added.
This is pulled from [Semantic Scholar's graph API](https://api.semanticscholar.org/api-docs/graph). Building out my citation tree
seemed like a bit much to start with, but that is a future project.

While the text can be used for certain search methods, it's not optimal for them all. Text needs to be chunked.
For now spans of a fixed word length were used. These lengths were 128,256,512 words. Originally I wanted all spans to be in one table. I noticed that query speed is directly 
related to the number of items. Even with an index, larger number of items are heavily penalized.
So for now every span, offset combination is in a separate table.

#### Indexing


Ok, so the data has been processed and downloaded. Next step is to make it searchable.

Earlier I buried the lead a bit. There aren't just 5 methods of searching, there are hundreds.
Beyond the high-level categories I outlined above, each broad category contains methods and other variants.
Methods can be applied to different fields such as the title, abstract, body, spans, etc..
There is also a question of how and when they are applied. Below is a table of combinations currently supported

| End Use Case   | Method Family | Technique / Model        | Field(s) Indexed          | Representation Type       | Scoring Type                          |
|----------------|--------------|--------------------------|---------------------------|----------------------------|----------------------------------------|
| Search         | Keyword      | BM25                     | title, abstract, body, spans    | Sparse (token-based)       | Explicit (BM25: TF-IDF variant)        |
| Search         | Vector       | SPLADE                   | abstract, spans           | Sparse (learned expansion) | Explicit (dot product)                 |
| Search         | Vector       | Nomic-1.5         | abstract, spans            | Dense                      | Explicit (cosine / dot product)        |
| Search         | Exact        | SQL equality / filters   | title         | Exact / structured         | Explicit (boolean / exact match)       |

The number of combinations becomes a bit absurd when you start actually
thinking. That's before we even start talking about combining methods through fusion
or reranking.


#### UI/UX

The front end is simplistic, both visually and in terms of structure. The landing page that provides 3 additional screens.

![Research hub landing page](/images/research_hub_homepage_site.png)


The search page is where the magic happens. Users can write queries up top.

![Research hub landing page](/images/research_management_tool_search_papers_site_empty.png)

By clicking on filters, users have the ability to select a large number of index + source combinations.

![Research hub landing page](/images/search_papers_results_site_dropdown_open.png)

The results are presented in cards that have a bookmark icon and a drop down that allows them to add questions to a given RQ.

### Curation

Ok so you like a search result, what next? 
I created two repositories. 
1. A library (I assume you are familiar with the concept)
2. A repository of research question


#### Library


Library is still pretty rudimentary, but acts as a store of papers I want to read or have read.

![Research hub landing page](/images/search_papers_results_site_dropdown_open.png)

It's currently limited to adding papers, removing papers, or adding them to RQs. Currently using it to triage 
papers until they find a more permanent home elsewhere. Eventually I'd like to have more filters and search over the 
library. This hasn't really been a high priority since I'm primarily using the tool to organize research questions.

#### Research Question Management

Research questions is the term used, but this is a place holder for any way of organizing results.
There is a landing page that has the research questions.

![Research hub landing page](/images/research_questions_list_site_active.png)

Upon clicking on a RQ, you can see the papers associated with that research question.
There are also recommendations on the right hand side.

![Research hub landing page](/images/research_question_attached_papers_site_empty.png)

There are two recommendation engines
1. vector based
2. bibliobased


![Research hub landing page](/images/research_question_sheets_site_recommendations_results_5.png)

Under the hood the methods are pretty simple:
1. Vector (Abstract)
	1. Pull in all papers for the RQ
	2. Aggregate the vectors for each paper (mean, max, etc..)
	3. Use that aggregate to find relevant papers
	4. Return top-k excluding the papers shared
2. Bibliographic
	1. Run bibliographic similarity across all papers in the RQ
	2. Use [RRF](https://www.paradedb.com/learn/search-concepts/reciprocal-rank-fusion) to fuse the rankings
	
I eventually want to add more, but this has worked well enough to start.

## Next Steps

This was a good first stab, but there are still a number of issues that need to be addressed.

### Large Scale Usage

I have 5 RQs I'm currently I've used this tool for. It's worked pretty well so far and surfaced  papers  I haven't found through the aforementioned tools. That being said, 5 RQs is not a lot and even the largest
only has 20 papers as of writing. I'm currently planning a large scale survey for definitions of domain in NLP which I expect to net 100+ papers so that will likely be a good test
of how the system performs at scale. Currently there isn't the ability to search and pagination was an after thought, so I expect the issues to be primarily related to the UI.

### Data Analysis and Data Gathering

I used a straightforward approach to create my dataset and monitor its health. The biggest shortcomings I see are the following

1. A single source was used (Arxiv), so some papers or other works are missing. It's also harder to create valid citations since the final venue isn't published.
2. The ingestion pipeline could use additional work
	1. Currently tags are used to filter relevant papers, but it's a naive heuristic. I'm in the process of training a model to score relevance. A first attempt didn't work reliably (a lot of false negatives), so I'm currently labelling a datset.
	2. The pipeline fails occasionally and there isn't really a mechanism to identify failure points or recover form them. This is tracked, but currently my system makes it hard to fix individual failures
	3. Pull in citations and some other information from outside sources, but it looks like some data may be missing due to inconsistencies between the format
	4. Structure isn't used to pull in text or chunk, so some information gets lost
3. There are no individual or corpus level quality checks
	1. At the individual level
		1. No checks for methodology
		2. Some preprints are very wonky, there is everything from class projects to rants
		3. Spans are sometimes irrelevant text and there isn't a tag or way of verify that
	2. Corpus level
		1. Dedupe
		2. What percentage of data is missing or isn't indexed

### UI/UX

The underlying premise is that it's hard to choose the right method, but part of the reason is 
that using the methods correctly isn't intuitive. Keyword methods require choosing keywords relevant
to your query and in your corpus. Neural methods are sensitive to word choice. Query augmentation
approaches should be corpus aware and reflect what words are being added to the user.
Each method has it's quirks.

How to interface with each method should be built into the UI. I think there are a couple ways of doing that
1. Search suggestion methods. Some of these are ideas, so it's unclear how they would help performance and this would need to be evaluated.
	1. For keyword methods suggest synonyms, other often used words using corpus statistics
	2. For neural methods use an n-gram LLM over the finetuning corpus to propose good searches (making sure the input resembles the training data as much as possible).
2. For query augmentation using the same mechanism as above and giving the user a way to understand if/how that's being used. There's a lot to unpack here, so tbd.
3. Use of an agent as a router to abstract method selection from the user. More on a potential mechanism for this in the next section.


### More Search Tools

The approaches used so far are limited in number and scope. Here are additional approaches I'd like to implement 
1. Cross encoder
2. Listwise prompt based reranker
3. Pair wise prompt based reranker
4. [ColBERT](https://arxiv.org/abs/2004.12832) and the subsequent works 
5. Additional dense embeddings, including custom dense embeddings that allow for different question formats (e.g., research questions -> spans)
6. Query rewriting/augmentation, e.g., [DocT5query](https://github.com/castorini/docTTTTTquery) or 
7. Document augmentation
8. PRF and other "corpus expansion" methods 

Some of these approaches are more complex than others to implement. ColBERT requires additional indexing not
available in paradedb. Most other methods can be supported with my current stack,
they just require additional code or training additional models. As we increase the number of tools,
the number of pipelines increases exponentially and we need to handle this in a straightforward way.


### DSL for Search

One of the things that's hard about search is that it isn't one single task,
it's a large myriad of tasks masquerading as one [like similarity](). A given query from different users
could have different valid results, different inputs could map to different
User intent is hard to decipher from just a query. Not all searches 
are as simple as just finding results, [like tip of the tongue search](https://arxiv.org/abs/2502.17776), exploration,
recommendation, and question answering for example.

Each task is currently optimized for independently. We have QA methods and embeddings, tip of the tongue systems, IR, etc..
Selecting the correct method has large value, as evidenced by these papers:
1. https://arxiv.org/pdf/2602.00296
2. https://arxiv.org/abs/2506.13743


You're not going to do QA and simple keyword retrieval with the same tool. For that reason intent needs to be understood and the users intent needs to be potentially clarified through direct interaction.

Rather than use an agent (or other system) to directly manipulate results, it might make more sense to use them with a better
tool that gives flexibility on the right tool(s) to use. 
Current LLMs can write code (to a certain extent), so why not have code to create search pipelines optimized for each problem.
A DSL (domain specific language) for search and recommendations would be a good opportunity for this. It provides 
composability, inpectability and reproducibility. 
A DSL can also be used as part of an RL pipeline (I think, lol RL is not my specialty) and in turn models could be trained 
to better use the tool.

Moreover a DSL can allow for a compact way of constructing and quickly trying out different pipelines without needing to start over from scratch.
A lot of the components could be recombined in new and different ways. A DSL can allow for experimentation and optimization in a way 
that isn't otherwise possible.

### Beyond Search

Beyond search, there are other techniques to find relevant papers. 
One idea I've been toying with is that we can use workflows to do systematic surveys over the entire corpus.

The approach is simple
1. define criteria for relevant paper that can be translated into tags/classes
2. design a workflow that uses simple conditional logic to filter results
3. implement pipeline
	1. implement skeleton for pipeline
	2. create initial classification prompts
	3. run pipeline
	4. review each classification stage to make sure that results are adequate
4. review the pipeline

This approach allows you to cast a wider net, estimate global performance (risk of non-inclusion)
using traditional metrics we are all familiar with. This likely isn't a new approach, but 
I think the trick here is how to make this approach work reliably and quickly. I'm currently trying 
this out for my survey approach.

## Closing Remarks

Overall I'm happy with the preliminary results. A bit meta, but as I was researching and verifying claims 
for this post I used to the tool a few times. For me this work solves the search and organization problems 
I was having with earlier versions of the tool.  
I plan on sharing the underlying code when it's further along. I haven't productionalized 
 the code and the front end is entirely vibe coded.
I'm currently using it to survey definitions of domains in NLP, so 
there will be more updates (and potentially a code base) after I get done with that.