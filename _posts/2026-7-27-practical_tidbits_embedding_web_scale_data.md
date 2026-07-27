# Practical Tidbits: Serving Web-scale Dense Embeddings

I haven't posted in a while. Life got busy, and the [larger scale characterization experiments are still running](https://felixlabelle.com/2026/05/15/large_scale_prompt_experiments.html).

In the meantime I've been participating in [TREC RAG 2026](https://trec-rag.github.io/). 
The corpus used, [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), has 400B tokens over 553 million documents.
Normally I work on model development or prototypes, so the scale of this dataset was a new challenge for me.

While there are a number of YouTube tutorials or white papers that deal with embedding corpora, most are done over smaller, well-known datasets. When it comes to large-scale dense embeddings 
there are a lot of different pieces that those tutorials don't put together for you. 
This blog post is meant to fill a gap in that space and provide an example 
of one way of embedding web-scale data and what I would do differently next time around.

## Context/Challenges of Web Scale

TREC RAG is a competition organized as part of [NIST](https://www.nist.gov/)'s [Text REtrieval Conference](https://trec.nist.gov/) (TREC). 
The 2026 track evaluates systems through two related tasks:
* the retrieval task asks participants to find and rank the most relevant segments for a given topic
* the full RAG task extends this by asking systems to generate an answer grounded in—and citing—the retrieved evidence

Participants can access the corpus through a hosted [PySerini API](https://github.com/castorini/pyserini) (BM25 index) or build their own retrieval system.
Before trying unique solutions for the retrieval portion of the task, we first need to establish a strong baseline.
I chose to use dense embeddings as that strong baseline primarily because they are well supported in terms of tooling and are a known quantity to me.
I've embedded datasets before, just nothing on this scale; making it work across 553 million documents poses additional challenges:
1. Disk space required; just the raw corpus is about 600 GB, indices are 2-6 TB depending on settings
2. Speed required to embed this many documents in a reasonable time frame
3. Dealing with concurrency and multi-machine setups
4. Serving an index that can't be fit entirely in RAM

## How the System Works

The end goal is to host a search API that takes in a query and returns the 
top-k documents and the relevant metadata.
The system has three jobs: 
1. preserve a stable mapping back to the original ClimbMix documents,
2. generate and store run-specific passage embeddings (passages are derived from the ClimbMix documents)
3. serve those embeddings through an index that can be queried at interactive speeds.

The API is the visible piece, but most of the work is in making sure IDs, metadata, chunks, and FAISS entries stay aligned across the preprocessing and indexing pipeline.
This requires infrastructure to embed documents and build out an index.

### Surface Level (API)

The API exposes three routes:
1. POST /{run_id}/search : returns top-k results for a given query over a run_id (think embedding, we'll get there) 
2. GET /corpus/{run_id}/{faiss_id} : returns a specific object
3. GET /corpus/{shard_id} : returns a given document

Below is a visualization of the routes provided:

<pre class="mermaid">
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif", "primaryColor": "#eff6ff", "primaryTextColor": "#172033", "primaryBorderColor": "#7aa2d6", "lineColor": "#5f6f89", "tertiaryColor": "#f8fafc"}}}%%
flowchart TD
    subgraph REQUESTS[Inputs]
        direction LR
        QUERY[/Query text/]
        LOOKUP[/run_id + faiss_id/]
        SHARD[/shard_id/]
    end

    subgraph API[Search API]
        direction TB
        SEARCH["POST /{run_id}/search<br/>Rank top-k matches for one embedding run"]
        FAISS_GET["GET /corpus/{run_id}/{faiss_id}<br/>Fetch a stored passage by FAISS id"]
        SHARD_GET["GET /corpus/{shard_id}<br/>Fetch the original Climbmix document"]
    end

    subgraph RESPONSES[JSON responses]
        direction LR
        RESULTS["SearchResult[]<br/>faiss_uid, shard_id, score, text"]
        OBJECT["RunTable row<br/>faiss_uid, shard_id, text"]
        DOCUMENT["ClimbmixCorpus row<br/>shard_id, text, source_file, source_row"]
    end

    QUERY -->|POST body| SEARCH
    LOOKUP -->|path params| FAISS_GET
    SHARD -->|path param| SHARD_GET

    SEARCH -->|ranked run rows| RESULTS
    FAISS_GET -->|run row| OBJECT
    SHARD_GET -->|corpus row| DOCUMENT

    classDef input fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px,color:#172033;
    classDef endpoint fill:#eff6ff,stroke:#2563eb,stroke-width:1.8px,color:#172033;
    classDef response fill:#ecfdf5,stroke:#059669,stroke-width:1.8px,color:#172033;
    class QUERY,LOOKUP,SHARD input;
    class SEARCH,FAISS_GET,SHARD_GET endpoint;
    class RESULTS,OBJECT,DOCUMENT response;
</pre>

The first route is primarily what we need, the others allow us to interface with 
PySerini more easily by allowing us to pull original documents or metadata that can be used 
to implement RRF/other methods.

### How the API (and other tooling) is Hosted

Beyond hosting the API there are other needs. Concretely we need to embed queries at run time,
bulk embed queries when creating the index, store the dense embedding index, and  store the metadata 
associated with the index.

Each of these needs requires different compute and code. The diagram below details the tools, hardware, and software required to index and serve the pipeline.

<pre class="mermaid">
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif", "primaryColor": "#eff6ff", "primaryTextColor": "#172033", "primaryBorderColor": "#7aa2d6", "lineColor": "#5f6f89", "tertiaryColor": "#f8fafc"}}}%%
flowchart LR
    USER["Client / search request"]

    subgraph GCP["Google Cloud Platform"]
        direction LR

        subgraph GPU["GPU machine(s)<br/>bulk embedding"]
            direction TB
            K8S["Kubernetes jobs"]
            CONTAINERS["Docker containers"]
            BULK["Bulk embedding workers"]

            K8S --> CONTAINERS
            CONTAINERS --> BULK
        end

        subgraph INDEX["Index host VM"]
            direction TB
            API["FastAPI search API"]
            FAISS[("FAISS vector indices")]
            POSTGRES[("PostgreSQL<br/>documents + metadata")]
        end

        subgraph ONLINE["CPU machine<br/>online serving"]
            direction TB
            EMBED["Online embedding service"]
        end
    end

    POSTGRES -->|documents for indexing| BULK
    BULK -->|document embeddings| FAISS

    USER -->|query| API
    API --> EMBED
    EMBED -->|query embedding| FAISS
    FAISS -->|matching document ids| API
    POSTGRES -->|documents + metadata| API
    API -->|ranked results| USER

    classDef external fill:#f8fafc,stroke:#64748b,stroke-width:1.6px,color:#172033;
    classDef orchestrator fill:#f5f3ff,stroke:#7c3aed,stroke-width:1.6px,color:#172033;
    classDef compute fill:#eff6ff,stroke:#2563eb,stroke-width:1.8px,color:#172033;
    classDef service fill:#ecfdf5,stroke:#059669,stroke-width:1.8px,color:#172033;
    classDef storage fill:#fff7ed,stroke:#ea580c,stroke-width:1.8px,color:#172033;
    class USER external;
    class K8S,CONTAINERS orchestrator;
    class BULK,EMBED compute;
    class API service;
    class FAISS,POSTGRES storage;
</pre>

As a TL;DR, there are three machines:
1. A CPU-only machine that embeds queries at run time
2. An index host that has the dense index, PostgreSQL and hosts the API. PostgreSQL was chosen because a DB makes persistence easier, a relational database makes aligning runs easier, and I'm familiar with the tool.
3. A K8 cluster of 16 L4s that runs when indexing the dataset

### Creating the Index
We first need to index the data. To this end we ingest ClimbMix into SQL, assign stable shard IDs, chunk each document for a given embedding run, 
write passage metadata to a run-specific table, build smaller FAISS shards in parallel, merge those shards, and serve the merged index through the API. 


[ClimbMix is downloaded from Hugging Face](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
which results in a large set of parquet files. The PySerini index 
has "shard_ids" that are generated. To map the data correctly, we need to associate the shard_ids with the rows in the parquet files. This is done using 
[a script created by the organizers](https://github.com/TREC-RAG/trec-rag-skills/blob/main/skills/trec-rag-climbmix-corpus-creation/scripts/create_climbmix_corpus.py) and the result is added 
to the dataset and stored in the SQL database in the 
"climbmix_corpus" table.

<pre class="mermaid">
%%{init: {"theme": "base", "themeVariables": {"fontFamily": "Inter, ui-sans-serif, system-ui, sans-serif", "primaryColor": "#eff6ff", "primaryTextColor": "#172033", "primaryBorderColor": "#7aa2d6", "lineColor": "#5f6f89", "tertiaryColor": "#f8fafc", "entityBkg": "#eff6ff", "entityBorder": "#2563eb", "attributeBackgroundColorOdd": "#f8fafc", "attributeBackgroundColorEven": "#ffffff"}}}%%
erDiagram
    CLIMBMIX_CORPUS {
        TEXT shard_id PK
        TEXT text "NOT NULL"
        TEXT source_file "NOT NULL"
        INT  source_row "NOT NULL"
    }

    RUNS {
        TEXT run_id PK
        JSON run_metadata "NOT NULL"
    }

    RUN_TABLE {
        TEXT faiss_uid PK
        TEXT shard_id FK
        TEXT text "NOT NULL"
    }

    CLIMBMIX_CORPUS ||--o{ RUN_TABLE : "chunks into"
    RUNS ||--o{ RUN_TABLE : "materializes"
</pre>

The key design choice here is separating corpus identity from run identity. 
`climbmix_corpus` is the stable source of truth: one row per original document with its `shard_id`, 
source file, row, and text. Each embedding run then materializes its own passage table keyed by FAISS IDs and linked back to the original 
`shard_id`. That split makes it possible to compare dense retrieval against Pyserini/BM25, 
run multiple embedding configurations without rewriting the source corpus, and keep each FAISS index smaller and easier to rebuild.


Ok now that we have the corpus in a SQL db, we can iterate over the table.
Each series of embeddings will be referred to as a "run". A run is a combination of embeddings, chunking settings, batch size, and a user-assigned run_id.
First we add the run metadata to the 'RUNS' table. Once the run is entered,
the model iterates over each row of the ClimbMix corpus table. 
Each row is chunked. The chunks are then batched together.
The batches are turned into embeddings. Each time a batch is completed it is added to 
a queue. The queue is then written to both the SQL and FAISS index. 
The metadata (text and a UID) for each passage is stored in a 
table that has the run name in it. The reason each run is separated is to have 
smaller indices (rather than having a single mega table).

To do this with enough speed a multi-producer, multi-consumer architecture was used.
 Documents are fed into a queue by each producer,
which is consumed BATCH_SIZE items at a time. Each batch is embedded. The resulting output is fed into another queue
consumed by the indexer. The indexer creates shards and writes the sharded metadata (with corresponding FAISS index IDs)
to the SQL database. Without threading, we can't leverage the Kubernetes cluster fully.
The indices themselves are using inner product, built with normalized vectors, using IVFPQ, and 
use explicit IDs to map entries back to the PostgreSQL database.

After the shards are written, they need to be merged.

<details markdown="1">
<summary> Example Merge Code </summary>

```python
from pathlib import Path

import faiss


def merge_faiss_shards(shard_dir: str, output_path: str) -> None:
    shard_paths = sorted(Path(shard_dir).glob("*.faiss"))
    if not shard_paths:
        raise ValueError(f"No FAISS shards found in {shard_dir}")

    merged = faiss.read_index(str(shard_paths[0]))

    for shard_path in shard_paths[1:]:
        shard = faiss.read_index(str(shard_path))
        merged.merge_from(shard, 0)  # keep existing FAISS IDs unchanged

    faiss.write_index(merged, output_path)


merge_faiss_shards("/path/to/shards", "/path/to/merged.faiss")
```

</details>
<br>


The utility code assumes each shard was built from the same trained index and that FAISS IDs are already globally unique.
Just be mindful that the merge takes a hot minute and is proportional to the amount of data and number of files present. With the full run it takes about 2-3 days on a 16 core machine, using ~20 GB 
of RAM at the peak with ~3k shards.

## Caveats/Lessons Learned

While I enjoyed the project, it was essentially the engineering equivalent of going to the Home Depot every step of a home reno.
Every time I made it further, I realized there was a problem I hadn't thought of. While small scale testing highlighted 
most issues, there are a couple I only discovered after the first full scale run.
I live in the metaphorical Home Depot at this point. Below are a couple of high-level takeaways to help avoid pain points I faced.
Below is a piece of code written after the fact that addresses some of the pain points below.
It is used to generate the results below.

<details markdown="1">
<summary> Caveat Utility Code </summary>

```python
import argparse
import math

from datasets import load_dataset
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm.auto import tqdm




def chunk(tokens, chunk_size):
    # CHUNK_SIZE -2 is to account for BOS EOS
    return [tokens[i:i + chunk_size - 2] for i in range(0, len(tokens),  chunk_size - 2)]

def estimate_num_chunks(doc_stats, char_token_ratio, chunk_size):
    num_chunks = 0
    for doc_stat in doc_stats:
        token_estimate = doc_stat['num_chars'] / char_token_ratio
        num_chunks += math.ceil(token_estimate / (chunk_size-2))
    return num_chunks
    
def estimate_unembedded_data(chunk_stats, char_token_ratio, chunk_size):
    lost_chars = 0
    total_chars = 0
    
    for chunk_stat in chunk_stats:
        if chunk_stat['token_char_ratio'] < char_token_ratio:
            lost_chars += (math.ceil(char_token_ratio * (chunk_size-2)) - chunk_stat['num_chars'])
        total_chars += chunk_stat['num_chars']
        
    return lost_chars/total_chars

unit_scale = {"KB" : 1_000, "GB" : 1_000_000_000, "MB" : 1_000_000, "TB" : 1_000_000_000_000}
def estimate_disk_size(num_chunks, sample_size, num_docs, num_dimensions, bytes_per_dim, units="TB"):
    num_bytes_estimate = (num_docs/sample_size) * num_chunks * num_dimensions * bytes_per_dim
    num_units_estimate = num_bytes_estimate/unit_scale[units]
    return f"{num_units_estimate} {units}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate char/token chunking tradeoffs.")
    parser.add_argument("--visualize-loss", action="store_true", help="Generate loss plots and print the loss table.")
    parser.add_argument("--sample-char-ratio", type=float, help="Show lost-text samples at this chars/token ratio.")
    parser.add_argument("--estimate-disk-size", action="store_true", help="Estimate index disk size.")

    # NOTE: Shared arguments for data loading
    parser.add_argument("--sample-buffer-size", type=int, default=1_000_000)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--target-model")

    parser.add_argument("--num-samples-to-show", type=int, default=10)
    parser.add_argument("--disk-char-ratio", type=float, help="Chars/token ratio to use for disk-size estimation.")
    parser.add_argument("--num-docs", type=int, help="Total number of documents in the full dataset.")
    parser.add_argument("--num-dimensions", type=int, help="Embedding dimensions per chunk.")
    parser.add_argument("--bytes-per-dim", type=int, help="Bytes stored per embedding dimension.")
    parser.add_argument("--disk-size-units", choices=sorted(unit_scale), default="TB")
    args = parser.parse_args()

    has_action = args.visualize_loss or args.sample_char_ratio is not None or args.estimate_disk_size
    if not has_action:
        print("Select at least one action flag: --visualize-loss, --sample-char-ratio, or --estimate-disk-size")
        raise SystemExit(0)

    required_args = ["chunk_size", "dataset", "target_model"]
    missing_args = [f"--{arg.replace('_', '-')}" for arg in required_args if getattr(args, arg) is None]
    if args.estimate_disk_size:
        for arg in ["disk_char_ratio", "num_docs", "num_dimensions", "bytes_per_dim"]:
            if getattr(args, arg) is None:
                missing_args.append(f"--{arg.replace('_', '-')}")
    if missing_args:
        parser.error(f"the following arguments are required for selected actions: {', '.join(missing_args)}")

    cm_dataset = load_dataset(args.dataset, split="train", streaming=True)
    cm_samples = [
        item['text']
        for item in cm_dataset.shuffle(buffer_size=args.sample_buffer_size, seed=args.random_seed).take(args.sample_size)
    ]
    doc_stats = [{"num_chars" : len(cm_sample)} for cm_sample in cm_samples]

    needs_chunk_stats = args.visualize_loss or args.sample_char_ratio is not None
    if needs_chunk_stats:
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)
        tokenized_samples = tokenizer(cm_samples, truncation=False, max_length=None,add_special_tokens=False,return_offsets_mapping=True)

        chunk_stats = []
        for doc_index, (cm_sample, tokenized_offsets) in enumerate(tqdm(zip(cm_samples, tokenized_samples['offset_mapping']))):
            chunk_offsets = chunk(tokenized_offsets,args.chunk_size)
            for chunk_index, chunk_offset in enumerate(chunk_offsets):
                start_idx, end_idx = chunk_offset[0][0], chunk_offset[-1][1]
                chunk_stats.append({
                    "doc_index" : doc_index,
                    "chunk_index" : chunk_index,
                    "start_idx" : start_idx,
                    "end_idx" : end_idx,
                    "num_chars" : end_idx - start_idx,
                    "num_tokens" : len(chunk_offset),
                    "token_char_ratio" : (end_idx - start_idx)/len(chunk_offset),
                })

    if args.visualize_loss:
        token_ratios = [chunk_stat["token_char_ratio"] for chunk_stat in chunk_stats]
        plt.figure(figsize=(10, 6))
        plt.hist(token_ratios, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel("Characters per token")
        plt.ylabel("Number of chunks")
        plt.title("Chunk Character-Per-Token Distribution")
        plt.tight_layout()
        plt.savefig("char_token_ratio_histogram.png", dpi=200)
        plt.show()

        # Serves as a reference to estimate size
        golden_num_chunks = len(chunk_stats)
        print("Gold truth num chunks: ", golden_num_chunks)
        char_ratios_to_try = [val/10 for val in range(10, 61,1)]

        results = []
        for char_ratio in char_ratios_to_try:
            chunk_count_pct = estimate_num_chunks(doc_stats, char_ratio, args.chunk_size) / golden_num_chunks * 100
            data_loss_pct = estimate_unembedded_data(chunk_stats, char_ratio, args.chunk_size) * 100
            results.append({
                "char_ratio": char_ratio,
                "chunk_count_pct": chunk_count_pct,
                "data_loss_pct": data_loss_pct,
            })

        print(f"{'char_ratio':>10} {'chunk_count_pct':>16} {'data_loss_pct':>14}")
        for result in results:
            print(f"{result['char_ratio']:>10.1f} {result['chunk_count_pct']:>16.2f} {result['data_loss_pct']:>14.2f}")

        chunk_count_percentages = [result["chunk_count_pct"] for result in results]
        data_loss_percentages = [result["data_loss_pct"] for result in results]

        plt.figure(figsize=(12, 7))
        plt.plot(chunk_count_percentages, data_loss_percentages, marker='o', linewidth=1.5, markersize=4)
        label_offsets = [(0, 8), (0, -12), (8, 0), (-8, 0)]
        for index, result in enumerate(results):
            plt.annotate(
                f"{result['char_ratio']:.1f}",
                (result["chunk_count_pct"], result["data_loss_pct"]),
                textcoords="offset points",
                xytext=label_offsets[index % len(label_offsets)],
                ha="center",
                va="center",
                fontsize=7,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.65},
            )
        plt.xlabel("Estimated chunks (% of exact chunking; lower means more compression)")
        plt.ylabel("Estimated data loss (%)")
        plt.title("Data Loss vs Compression")
        plt.grid(True, linestyle='--', alpha=0.35)
        plt.margins(x=0.04, y=0.08)
        plt.tight_layout()
        plt.savefig("data_loss_vs_compression.png", dpi=200)
        plt.show()

    if args.sample_char_ratio is not None:
        sample_char_budget = math.ceil(args.sample_char_ratio * (args.chunk_size - 2))

        # NOTE: Separate into a function that takes docs, chunk_stats
        print(f"\nLost data samples at {args.sample_char_ratio:.1f} chars/token:")
        shown_samples = 0
        for chunk_stat in chunk_stats:
            if chunk_stat["token_char_ratio"] >= args.sample_char_ratio:
                continue

            cm_sample = cm_samples[chunk_stat["doc_index"]]
            lost_start_idx = chunk_stat["end_idx"]
            lost_end_idx = min(chunk_stat["start_idx"] + sample_char_budget, len(cm_sample))
            unembedded_text = cm_sample[lost_start_idx:lost_end_idx]
            if not unembedded_text.strip():
                continue

            shown_samples += 1
            print("\n" + "=" * 80)
            print(
                f"Sample {shown_samples}: doc={chunk_stat['doc_index']} "
                f"chunk={chunk_stat['chunk_index']} "
                f"chars/token={chunk_stat['token_char_ratio']:.2f} "
                f"unembedded text={len(unembedded_text)}"
            )
            print("Unembedded text:")
            print(unembedded_text)

            if shown_samples == args.num_samples_to_show:
                break

        if shown_samples == 0:
            print("No chunks with displayable unembedded text found.")

    if args.estimate_disk_size:
        estimated_num_chunks = estimate_num_chunks(doc_stats, args.disk_char_ratio, args.chunk_size)
        disk_size = estimate_disk_size(
            estimated_num_chunks,
            len(doc_stats),
            args.num_docs,
            args.num_dimensions,
            args.bytes_per_dim,
            args.disk_size_units,
        )
        print(f"Estimated chunks in sample: {estimated_num_chunks}")
        print(f"Estimated disk size: {disk_size}")
```

</details>
<br>

### Chunking

The documents provided vary in size from less than a hundred tokens to a million tokens.
Due to the context size limits of embedding models and practical limits about how much data they 
can accurately represent (there is an information-theoretical limit for lossless representation, but getting an accurate estimate is hard so just trust me bro)
we chunk data. 

I assume we're all aware of chunking. It could be as simple as using whitespace, regexes, etc.
With a dataset of this size the most important part is going significantly faster than the embedding speed so it doesn't become a choke point.
Chunkers are not complicated to implement, but to save time and provide different options I wound up using [Chonkie](https://docs.chonkie.ai/common/welcome).


I first tried the [Token Chunker](https://docs.chonkie.ai/oss/chunkers/token-chunker#token-chunker),
but it was too slow. I suspect this was [due to the use of HF tokenizers, which are comparatively slow to other options](https://krabarena.com/claims/gigatoken-ran-26-2x-faster-than-tiktoken-on-a-174-mb-gpt-2-owt-tokenizer-slice?utm_source=chatgpt.com).
Speeds were reduced by about 10x compared to just yeeting whole documents and the CPU/RAM usage was quite a bit higher (based on a few glances at htop, we only employ peak engineering/science here)
The [semantic chunker](https://docs.chonkie.ai/common/open-source#semanticchunker) was a little slower than the Token Chunker,
but not as much as I'd expect given that it runs a model.
While the [Sentence Chunker](https://docs.chonkie.ai/oss/chunkers/sentence-chunker) was the fastest of the three,
it still reduced the throughput possible due to how I have the code set up (by about half compared to whole documents).

I wound up using the [Fast Chunker](https://docs.chonkie.ai/oss/chunkers/fast-chunker#fast-chunker). 
It can split on delimiters, but the limitation of this chunker is that you need to pick the max number of chars, not tokens, as the cutoff point.
Therefore you need to pick a char cutoff that won't exceed the context of the model.
I set the limit to $$3.5 * max\_num\_tokens$$ characters. 3.5 is a 
very rough estimate of the number of chars in a token. 
The estimated number of chars in a token is important to get right. If it's too large, there is a risk of having too much text and losing some context.
If it is too small, we wind up with many embeddings and wind up greatly increasing
index size. Given the index size is already 3-4 TB, increasing it further seemed like a poor idea (I want multiple indices 
on the same machine).

I chose 3.5 knowing some information might be lost. I sampled a small number of documents 
and eyeballing the risk it seemed acceptable to me, but your mileage may vary. Below is a 
plot showing the tradeoffs between picking different character/token ratios. (There is also a table 
version because there always should be). One caveat is that I chunked naively in this example,
and if you use delimiters these numbers will be different. Just make sure you use your target 
tokenizer with the correct settings.


![Data loss versus chunk compression ratio](/images/data_loss_vs_compression.png)

<details markdown="1">
<summary>Data Loss Table</summary>

| Char/token ratio | Chunk count % | Data loss % |
| ---: | ---: | ---: |
| 1.0 | 349.79 | 0.00 |
| 1.1 | 320.22 | 0.01 |
| 1.2 | 295.97 | 0.01 |
| 1.3 | 275.21 | 0.01 |
| 1.4 | 257.57 | 0.01 |
| 1.5 | 242.45 | 0.01 |
| 1.6 | 228.51 | 0.02 |
| 1.7 | 216.60 | 0.02 |
| 1.8 | 206.43 | 0.03 |
| 1.9 | 197.37 | 0.03 |
| 2.0 | 188.98 | 0.04 |
| 2.1 | 180.58 | 0.06 |
| 2.2 | 173.40 | 0.07 |
| 2.3 | 167.10 | 0.09 |
| 2.4 | 161.23 | 0.11 |
| 2.5 | 156.35 | 0.13 |
| 2.6 | 152.08 | 0.22 |
| 2.7 | 148.19 | 0.26 |
| 2.8 | 144.28 | 0.31 |
| 2.9 | 140.33 | 0.39 |
| 3.0 | 136.36 | 0.44 |
| 3.1 | 132.21 | 0.58 |
| 3.2 | 128.43 | 0.69 |
| 3.3 | 124.86 | 0.84 |
| 3.4 | 121.72 | 1.01 |
| 3.5 | 118.77 | 1.27 |
| 3.6 | 116.11 | 1.60 |
| 3.7 | 113.92 | 1.99 |
| 3.8 | 111.98 | 2.55 |
| 3.9 | 110.28 | 3.26 |
| 4.0 | 108.62 | 4.26 |
| 4.1 | 107.05 | 5.78 |
| 4.2 | 105.62 | 7.73 |
| 4.3 | 104.37 | 10.17 |
| 4.4 | 103.19 | 13.19 |
| 4.5 | 102.04 | 17.11 |
| 4.6 | 100.91 | 21.83 |
| 4.7 | 99.73 | 26.87 |
| 4.8 | 98.63 | 32.13 |
| 4.9 | 97.49 | 37.56 |
| 5.0 | 96.41 | 42.87 |
| 5.1 | 95.38 | 48.48 |
| 5.2 | 94.41 | 53.62 |
| 5.3 | 93.44 | 58.31 |
| 5.4 | 92.49 | 62.88 |
| 5.5 | 91.38 | 67.19 |
| 5.6 | 90.28 | 71.27 |
| 5.7 | 89.16 | 75.00 |
| 5.8 | 87.94 | 78.55 |
| 5.9 | 86.73 | 81.97 |
| 6.0 | 85.48 | 85.29 |

</details>
<br>



If you pick a value of 1, you are never going to have unembedded data.
However, you will massively overestimate the number of chunks (by about 350%).
A value of 6 leads to 85% of the data not being embedded, so yeah.
In hindsight I would pick a value of 2.5, that seems like the best compromise
between size and unembedded text. 

One thing I would note is chunks with smaller ratios are, IMO,
more likely to be noise. The logic goes as follows, sections that are 
heavily tokenized are likely to be rare words or formatted text.  Below are some samples:

<details markdown="1">
<summary>Unembedded Data Examples at 2.5 & 3.5 Chars/Token</summary>

```text


Unembedded data samples at 3.5 chars/token:

================================================================================
Sample 1: doc=430 chunk=0 chars/token=2.33 unembedded text=598
Unembedded text:
Id":"1836454364553645636457"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description","A language of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1836458"},["text","EN"]]]],["element",{"elementId":"51"},["name","Type"],["description","The nature or genre of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1836459"},["text","NEWS"]],["elementText",{"elementTextId":"1836460129686","public":"1","featured":"1"},["fileContainer",["file",{"fileId":"27681"},["src"," Text"],["description"],["elementContainer",["element",{"

================================================================================
Sample 2: doc=430 chunk=1 chars/token=2.18 unembedded text=674
Unembedded text:
TextContainer",["elementText",{"elementTextId":"182450124502"},["text","2010-11-24"]]]],["element",{"elementId":"48"},["name","Source"],["description","A related resource from which the described resource is derived"],["elementTextContainer",["elementText",{"elementTextId":"1824503"},["text","Santa Cruz Sentinel"]]]],["element",{"elementId":"49"},["name","Subject"],["description","The topic of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1824504"},["text","Capitola Mall"]],["elementText",{"elementTextId":"1824505245062450724508"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description","A language of the resource

================================================================================
Sample 3: doc=430 chunk=2 chars/token=2.18 unembedded text=671
Unembedded text:
",["element",{"elementId":"94"},["name","Text"],["description"],["elementTextContainer",["elementText",{"elementTextId":"19139821188511871"},["text","CF-53021"]]]],["element",{"elementId":"50"},["name","Title"],["description","A name given to the resource"],["elementTextContainer",["elementText",{"elementTextId":"1811872"},["text","'Game-changer' Hundreds show up for Target store's opening at Capitola Mall"]]]],["element",{"elementId":"39"},["name","Creator"],["description","An entity primarily responsible for making the resource"],["elementTextContainer",["elementText",{"elementTextId":"181187311874"},["text","2012-07-25"]]]],["element",{"elementId":"48"},["name

================================================================================
Sample 4: doc=430 chunk=3 chars/token=2.29 unembedded text=619
Unembedded text:
11881"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description","A language of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1811882"},["text","EN"]]]],["element",{"elementId":"51"},["name","Type"],["description","The nature or genre of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1811883"},["text","NEWS"]],["elementText",{"elementTextId":"1811884128223","public":"1","featured":"1"},["fileContainer",["file",{"fileId":"27689"},["src"," Text"],["description"],["elementContainer",["element",{"elementId":"94"},["name","Text"],["descript

================================================================================
Sample 5: doc=430 chunk=4 chars/token=2.16 unembedded text=685
Unembedded text:
elementTextId":"180319503196"},["text","2013-06-07"]]]],["element",{"elementId":"48"},["name","Source"],["description","A related resource from which the described resource is derived"],["elementTextContainer",["elementText",{"elementTextId":"1803197"},["text","Santa Cruz Sentinel"]]]],["element",{"elementId":"49"},["name","Subject"],["description","The topic of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1803198"},["text","Capitola Mall"]],["elementText",{"elementTextId":"1803199"},["text","Shopping Centers"]],["elementText",{"elementTextId":"1803200032010320203203"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description

================================================================================
Sample 6: doc=430 chunk=5 chars/token=2.18 unembedded text=675
Unembedded text:
," Text"],["description"],["elementContainer",["element",{"elementId":"94"},["name","Text"],["description"],["elementTextContainer",["elementText",{"elementTextId":"1913996element17052551705240"},["text","CF-45646"]]]],["element",{"elementId":"50"},["name","Title"],["description","A name given to the resource"],["elementTextContainer",["elementText",{"elementTextId":"1705241"},["text","Target Says Yes to Capitola"]]]],["element",{"elementId":"41"},["name","Description"],["description","An account of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1705242"},["text","Target1705243"},["text","Fridy, Linda05244"},["text","2010-09-14"]]]],["element

================================================================================
Sample 7: doc=430 chunk=6 chars/token=2.21 unembedded text=657
Unembedded text:
"]],["elementText",{"elementTextId":"17052481705249"},["text","2010s"]]]],["element",{"elementId":"45"},["name","Publisher"],["description","An entity responsible for making the resource available"],["elementTextContainer",["elementText",{"elementTextId":"17052501705251"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description","A language of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1705252"},["text","EN"]]]],["element",{"elementId":"51"},["name","Type"],["description","The nature or genre of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1705253"},["text","NEWS"]],["

================================================================================
Sample 8: doc=430 chunk=7 chars/token=2.28 unembedded text=624
Unembedded text:
"},["text","CF-45297"]]]],["element",{"elementId":"50"},["name","Title"],["description","A name given to the resource"],["elementTextContainer",["elementText",{"elementTextId":"1699932"},["text","Captiola approves plans for Target store at mall"]]]],["element",{"elementId":"41"},["name","Description"],["description","An account of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1699933"},["text","Target"]]]],["element",{"elementId":"39"},["name","Creator"],["description","An entity primarily responsible for making the resource"],["elementTextContainer",["elementText",{"elementTextId":"16999349

================================================================================
Sample 9: doc=430 chunk=8 chars/token=2.27 unembedded text=625
Unembedded text:
,{"elementTextId":"1699943"},["text","EN"]]]],["element",{"elementId":"51"},["name","Type"],["description","The nature or genre of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1699944"},["text","NEWS"]],["elementText",{"elementTextId":"16999450905","public":"1","featured":"1"},["fileContainer",["file",{"fileId":"27685"},["src"," Text"],["description"],["elementContainer",["element",{"elementId":"94"},["name","Text"],["description"],["elementTextContainer",["elementText",{"elementTextId":"1913972169681696829"},["text","CF-45083"]]]],["element",{"elementId":"50"},["name","Title"],["description

================================================================================
Sample 10: doc=430 chunk=9 chars/token=2.20 unembedded text=665
Unembedded text:
,["elementTextContainer",["elementText",{"elementTextId":"16968321696833"},["text","2011-04-02"]]]],["element",{"elementId":"48"},["name","Source"],["description","A related resource from which the described resource is derived"],["elementTextContainer",["elementText",{"elementTextId":"1696834"},["text","Santa Cruz Sentinel"]]]],["element",{"elementId":"49"},["name","Subject"],["description","The topic of the resource"],["elementTextContainer",["elementText",{"elementTextId":"1696835"},["text","Capitola Mall"]],["elementText",{"elementTextId":"1696836"},["text","Capitola"]],["elementText",{"elementTextId":"1696837"},["text","Redevelopment"]]]],1696838"},["t
Estimated chunks in sample: 21864
Estimated disk size: 4.9523884032 TB


Unembedded data samples at 2.5 chars/token:

================================================================================
Sample 1: doc=430 chunk=0 chars/token=2.33 unembedded text=88
Unembedded text:
Id":"1836454364553645636457"},["text","Text"]]]],["element",{"elementId":"44"},["name","

================================================================================
Sample 2: doc=430 chunk=1 chars/token=2.18 unembedded text=164
Unembedded text:
TextContainer",["elementText",{"elementTextId":"182450124502"},["text","2010-11-24"]]]],["element",{"elementId":"48"},["name","Source"],["description","A related re

================================================================================
Sample 3: doc=430 chunk=2 chars/token=2.18 unembedded text=161
Unembedded text:
",["element",{"elementId":"94"},["name","Text"],["description"],["elementTextContainer",["elementText",{"elementTextId":"19139821188511871"},["text","CF-53021"]]

================================================================================
Sample 4: doc=430 chunk=3 chars/token=2.29 unembedded text=109
Unembedded text:
11881"},["text","Text"]]]],["element",{"elementId":"44"},["name","Language"],["description","A language of th

================================================================================
Sample 5: doc=430 chunk=4 chars/token=2.16 unembedded text=175
Unembedded text:
elementTextId":"180319503196"},["text","2013-06-07"]]]],["element",{"elementId":"48"},["name","Source"],["description","A related resource from which the described resource is

================================================================================
Sample 6: doc=430 chunk=5 chars/token=2.18 unembedded text=165
Unembedded text:
," Text"],["description"],["elementContainer",["element",{"elementId":"94"},["name","Text"],["description"],["elementTextContainer",["elementText",{"elementTextId":"

================================================================================
Sample 7: doc=430 chunk=6 chars/token=2.21 unembedded text=147
Unembedded text:
"]],["elementText",{"elementTextId":"17052481705249"},["text","2010s"]]]],["element",{"elementId":"45"},["name","Publisher"],["description","An ent

================================================================================
Sample 8: doc=430 chunk=7 chars/token=2.28 unembedded text=114
Unembedded text:
"},["text","CF-45297"]]]],["element",{"elementId":"50"},["name","Title"],["description","A name given to the resou

================================================================================
Sample 9: doc=430 chunk=8 chars/token=2.27 unembedded text=115
Unembedded text:
,{"elementTextId":"1699943"},["text","EN"]]]],["element",{"elementId":"51"},["name","Type"],["description","The nat

================================================================================
Sample 10: doc=430 chunk=9 chars/token=2.20 unembedded text=155
Unembedded text:
,["elementTextContainer",["elementText",{"elementTextId":"16968321696833"},["text","2011-04-02"]]]],["element",{"elementId":"48"},["name","Source"],["descr
Estimated chunks in sample: 28783
Estimated disk size: 6.5196027904 TB

```



</details>
<br>


Most are formatted text. While there are other examples that do include full text,
IMO not all the unembedded text is a major loss. Nonetheless
I just showed you a clever way of doing something stupid.
Estimating tokens based on chars 
is sloppy by its nature and should be avoided if possible. 
A char-based approach will inherently be faster just due to the simplicity of it. 
If you need speed above all else, use the technique above.
Otherwise I'd recommend just using a faster tokenizer and implementing a chunker by hand;
I opted not to since the baseline was a means to an end, not a goal (aka bigger fish to fry).

Beyond the nominal size, there is a question of which delimiter to use. Chonkie's 
docs recommend different patterns, but I found that using whitespace worked best
for two reasons 
1. Sizes are more consistent, which helps keep index size consistent by not having very tiny chunks
2. Size being more consistent also reduces the number of small chunks with no content. I did a smaller run with the "sentence" delimiter 
and found that these tiny chunks would often be my top results. If there was a question similar to the query it would 
wind up being matched to the query.

The notable downside is the risk of splitting up sentences, but the effect of this on performance is unclear.
It would require more study, again beyond the scope of this post.



### Embedding at Scale

The biggest choke point in terms of speed is embedding chunks.
We have 553 million documents. When I originally measured, I was getting 
about 100 embeddings per second using an A100 with 40 GB of RAM using 
[BGE 1.5 large](https://huggingface.co/BAAI/bge-large-en-v1.5). 
This model was chosen because of its relative size and relatively good 
performance.
 Assuming only 1 passage per document (which is incredibly naive)
this works out to about 2 months of time required to embed the entire corpus.
Realistically we will have between 600-1.2 Billion chunks to embed depending on 
chunk sizes so the time may be as much as 4 months at that rate.
More chunks means slower so that's where the trade-off for unembedded data comes into play.

I calculated that at least 1-2k embeddings per second are needed to finish in a matter of days or weeks.
To do it with sufficient speed we only have a couple of options:
1. Smaller models 
2. Scaling embedding

I chose to scale compute. While smaller models are an option,
the speed required would limit us to [FastText](https://fasttext.cc/) or [Potion models](https://huggingface.co/collections/minishlab/potion).
While these are good options, 
ultimately I wanted a bit more liberty when picking different embedding methods, namely in terms of performance.


I scaled up in 2 ways:
1. Scaling through Kubernetes and picking a suitable "node". I chose to use L4s due to a higher quota and not needing a lot of GPU VRAM for embedding models.
2. Using a thread pool with a producer-consumer model (described above) to make concurrent calls

One thing I would change about the design is how data is loaded. I read in data from SQL using bulk reads.
I then would iterate over the items one at a time and chunk them from there. While this worked fine,
I can't help but believe there is a better way. I think reading in documents this way 
created a choke point and jitter with reading speeds. I would notice the pipeline choke up from time to time.


### Index Size and Type 

Ok, so now we can embed things with enough speed! Cool cool cool. Next step is to index those embeddings.

The index design choices ended up mattering more than I expected.
FAISS indices are built in a modular fashion and frankly can be a bit confusing at first.
FAISS stores vectors and searches for nearby vectors under a chosen distance function, such as inner product or L2 distance.
The distance itself is represented as an index. That said there is a recursive nature to FAISS indices.
In other words an index can be made up of other indices which each add properties. Whether it be the distance used, quantization, use of IDs, use 
of disk instead of RAM, etc.. 
The easiest type of index to use in FAISS is a FLAT, in memory index. While an in-memory FAISS index is convenient, it is not realistic once the run is measured in billions of vectors and multiple terabytes.
 That pushed me from the onset towards a disk-backed IVF-style setup.
IVF is a cluster-driven index, grouping vectors into clusters.

Below is an example of how to create a FAISS index like the one used:

<details markdown="1">
<summary>Minimal FAISS index stacking example</summary>

```python
import faiss
import numpy as np

dim = 1024
nlist = 1024  # use more lists at corpus scale

train_vectors = np.random.randn(50_000, dim).astype("float32")
vectors = np.random.randn(10_000, dim).astype("float32")
ids = np.arange(vectors.shape[0], dtype="int64")

quantizer = faiss.IndexFlatIP(dim)
ivfpq = faiss.IndexIVFPQ(
    quantizer,
    dim,
    nlist,
    64,  # PQ subvectors
    8,   # bits per code
    faiss.METRIC_INNER_PRODUCT,
)
index = faiss.IndexIDMap2(ivfpq)

# Conceptually: IndexIDMap2(IndexIVFPQ(IndexFlatIP))
index.train(train_vectors)
index.add_with_ids(vectors, ids)
```


</details>
<br>

`IndexFlatIP` is the coarse quantizer (inner product is the distance, normalized vectors make this act
like cosine similarity), `IndexIVFPQ` adds inverted lists and product quantization, and `IndexIDMap2` stores stable IDs that can be mapped back to SQL rows.
 For IVF-style indices, FAISS first needs a representative sample of embeddings so it can learn the coarse clusters used during search. 
 This means embedding a subset of the corpus, training the index on that sample, and only then streaming the full corpus through the trained index.
Training an index requires a sample large enough to represent the corpus, based on the number of clusters.

Since the index is running off disk, the number of clusters matters. Too many clusters requires
a large training set, which needs to all be stored in RAM. A 128k index requires at least 5M examples, which means ~40 GB of RAM
is used. This estimate is based on the runs I saw and again monitoring them through htop. A smaller index will result in slow run times.

Word to the wise, you need to estimate cluster size before committing to an index layout. 
The rough calculation is `num_vectors / num_clusters`, and at this corpus size 1024 clusters leaves each cluster far too large (nearly a million).
Moving to 64k or 128k clusters gave FAISS much smaller candidate buckets and made disk-backed retrieval acceptable for serving.
Coordinate the number of clusters based on the number of chunks you expect to have. The utility code should help you estimate the index size.  I now would estimate the target number of vectors, choose the IVF cluster count, sample enough embeddings to train that layout, and validate query latency before committing to the full run. 
 If the number of clusters is too small, each cluster becomes huge and retrieval slows down. 
 If the training sample is too small or unrepresentative, the clusters are worse and search quality or speed can suffer. Some clusters wind up 
 rather large and increase the variability in retrieval speeds. Lol there's no winning, it's engineering.

Another note is that the number of shards writing during embedding is also important. Originally I had small shards (~40 MB), meaning I wound up with 10s of thousands 
of shards. This caused the merge script to take an eternity if it completed at all. Make sure to keep the number of shards under 10k,
ideally 1-4k. The trade-off here is the larger the shards, the more RAM your machine needs at run time. If you are using a larger RAM machine, that may 
be worth the trade off.

### Machine Types

Another consideration is how to deploy the different components of this system.
Different tasks require different machines:

1. Bulk embedding -> cheap GPUs. I picked L4s because our availability was high enough and based on some quick testing the cost/throughput ratio was the best
2. One-off embeddings (i.e., inference time compute) -> AVX-512 machine (Intel Sapphire Rapids or higher). C3 8 core standard came clutch in here. If you have more models you may want more cores 
to distribute models more evenly across cores and to keep good speeds. You could use Kubernetes too to leverage the scaling. If you use very large decoder models for embeddings you may want to look 
at still using GPUs
3. Indexing.
	1. Training an IVF index for clustering is more involved and requires (AFAIK) having a lot of memory. I would recommend high-memory machines (64 GB+, ideally 256).
	This step is not as time consuming so you could swap out the machine for a smaller one when embedding.
	2. For the steady state a smaller machine can be used (8 cores, 16 GB RAM is fine). I recommend training and then swapping just because of the cost difference and duration
	of the indexing. If you are "[big ballin'](https://www.youtube.com/watch?v=xBq-H6pOHAE)" disregard this and just use the same machine as training the index.
4. Serving. I did it on the same machine as indexing, but really out of simplicity. You could likely get away with a smaller machine.


## Conclusion


Dense embeddings at this scale are feasible, but the hard parts are systems problems rather than modeling problems.
To be fair, some could be tackled as a modeling problem, but that was outside the scope of this post.
Ultimately the disk footprint, chunking throughput, chunking strategy, ID alignment, FAISS configuration, and serving architecture determine whether the system is usable.
 If I were doing this again, I would estimate index size and cluster layout earlier (using the code shared), make the corpus/run ID mapping explicit from the start, and 
 treat chunking throughput as a first-class bottleneck rather than a preprocessing detail.
Hopefully this is helpful if you ever need to embed a large corpus.
