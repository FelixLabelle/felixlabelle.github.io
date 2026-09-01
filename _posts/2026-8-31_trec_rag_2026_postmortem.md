# Post Mortem on TREC RAG 2026

TREC RAG was my first time competing in an IR/ML event.
While I'm not expecting us to place among the top groups, I think we'll do OK.
That aside, this blog is about the experience and some lessons for the next time.

## Our TREC RAG 2026 Experience (in a Nutshell)

TREC RAG is a competition organized as part of [NIST](https://www.nist.gov/)'s [Text REtrieval Conference](https://trec.nist.gov/) (TREC). The 2026 track evaluates systems through two related tasks:

* the retrieval task asks participants to find and rank the most relevant segments for a given topic
* the full RAG task extends this by asking systems to generate an answer grounded in, and citing, the retrieved evidence

This makes the competition interesting because retrieval and generation can be evaluated separately, but the full task still depends on both working together.
As a side project at work we put together a two-person team. We split the work between retrieval (spoilers, I did retrieval) and generation.
This worked well; each of us could focus on one part of the problem while sharing a relatively simple interface between the two.
For eval we relied on the framework provided by the organizers (more on it below).

Our approach was straightforward, we tuned and evaluated different systems over the [development set](https://github.com/TREC-RAG/trec-rag-data/tree/main/trec-rag-2026/development-data) using the [metrics](https://github.com/castorini/RAGDoll) provided by
the organizers of the competition. We performed a grid search over the following "elements":
1. Different query methods
	1. Query augmentation
        1. HyDE
        2. KW augmentation
	2. Retrieval method (BM25 was the only that worked, the vector index was incomplete)
	3. Fusion of different methods (RRF only)
	4. Use of a reranker (pointwise and listwise)
2. Different methods for chunking documents before being summarized
	1. Providing whole documents to the summarizer
	2. Generation from passages provided by the retrieval pipeline
	3. Having the summarization model predigest aselect passages
3. Different generative approaches
	1. Model
	2. Reasoning (except for Sol where this wasn't explored due to limits)

The metrics used to select models for submission were the synthetic qrels, factuality, and model arena results.
In the end what worked best for retrieval was keyword augmented BM25 + 3 synthetic documents concatenated to the query and in turn([HyDE](https://arxiv.org/abs/2212.10496)) fed into BM25. Results of the initital rankers were fused using RRF. One thing worth noting is that we tried passage retrieval and reranking, but both hurt performance. Subjectively it looked like 
the reranker faired poorly when given a large number of examples. While it mostly worked, some very 
informative chunks got placed lower than filler text from the same document. As a consequence passage based retrieval did significantly worse when it comes 
to the QRELs and our internal testing.

For generation we then took the best retrieval pipeline and paired it with what worked best for generation,
5.6 Sol on medium reasoning. The biggest improvement on the generation side came from 5.6 Sol.
The trend we noticed was recent models are noticeably better at following evidence and producing factual summaries. Model choice rather than reasoning 
was the biggest differentiator and 5.6 sol was particularly good.

## What Went Well

I'll start with the positives.
The TREC organizers did a good job. The track included development data, baselines, and tooling that made the competition approachable despite the size of the corpus and task. They were active in their slack channel and did a good job aiding participants IMO.

The team structure also worked well. With only two people we were small, but nimble.
Splitting retrieval and generation gave us clear ownership and let us make progress in parallel without introducing too much coordination overhead.
We tried a reasonable number of ideas and submitted a working end-to-end system. We'll see how well it actually performed once the official results are released.

## What Could Have Gone Better

This section is more about me personally than the team as a whole.

I was overly focused on improving individual components namely [dense embedding retrieval](https://felixlabelle.com/2026/07/27/practical_tidbits_embedding_web_scale_data.html), but frankly it took me much longer than I care to get working properly (basically after the competition). while I did quickly put together a prototype and baselines,
I didn't really spend much time improving results until 2 weeks before the competition's end as the embeddings ate most of my limited time.
Tangentially, I didn't split the pipeline into retrieval and generation stages early enough. This was a question of convience, but limited development speeds since we needed to rerun everything for each generation rather than use cached retrieval results.

Finally, it is unclear how reliable our development strategy was. Using a development split can still lead to overfitting.
The official results will be the real test of whether the selected pipeline generalized. Moreover I want to 
better understand evaluation of modern IR systems. The [evaluation tool provided by the organizers](https://github.com/castorini/RAGDoll) served us well,
but it would behoove me to understand evals in the post annotator world we live in.
I also didn't inspect as many examples as I would have liked. Aggregate metrics help compare runs, but they don't explain why a system works or fails.
More manual inspection would have helped identify potential issues with the pipeline and potentially other methods to explore.

## Closing Remarks

My main takeaways are
1. get end-to-end results faster
2. keep running end-to-end results throughout the process
3. define the pipeline boundaries early
4. manually inspect more examples
5. recent LLMs have improved when it comes to factuality. Retrieval still matters, citations still need to be checked, and the pipeline still needs to be evaluated as a whole.


Overall, this was a fun first competition. Once the official results are available, I'll have a better idea of which technical decisions worked, what didn't, and more importantly what to try next time.
