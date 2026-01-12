# 2025 -> 2026

2025 was a good year, but I didn’t accomplish as much as I initially expected. So is life. The reduced output is likely due to side quests that didn’t result in any artifacts (yet).

## Accomplishments

My research interests the last few years has circled around the inability to plan ML projects.
Beyond interpretability, we don't have the tools to characterize models or tasks and understand what 
is required to solve a given task.
I set out a goal last year to focus my work on having a better understanding of models and how hyper-parameters effect them. Here's what I blogged in that vein:

* **[“Practical Tidbits: Taking a Magnifying Glass to (Text) Classifier Performance”](https://felixlabelle.com/2025/06/15/evaluation_101.html)** was my attempt to formalize an approach to debugging text classifiers.
* **[“Temperature, Tokens, and Long Tales/Tails”](https://felixlabelle.com/2025/10/19/relationship_temp_token_gen.html)** focused on generation dynamics: how decoding choices interact with long-tail behavior, both in terms of output diversity and error modes. This was a fun little piece and will likely have a follow up this year.
* **[“On the Theoretical Limitations of Embedding-Based Retrieval”](https://felixlabelle.com/2025/11/21/review_of_theoritical_limit_ir.html)**. It was an analysis and critique of a paper that claimed there were important limits to embeddings. This was the most theory-heavy piece of the year, and it took a fair amount of time.


My other posts were more method focused, in other words applications of NLP (primarily LLMs) rather than studying higher level settings.
* **[“Using Landmarks to Extract Spans with Prompting”](https://felixlabelle.com/2025/08/28/span_extraction.html)** describes how to use structured anchors could make span extraction more reliable without reverting to a trained sequence labeling model. 
* **[“LLMs for Text Normalization (i.e., Domain Adaptation)”](https://felixlabelle.com/2025/12/25/word_replacement.html)** discusses a simple method to normalize text and how that relates to improved model performance
* **["LLMs for ETL"](https://felixlabelle.com/2025/02/12/wip_llms_for_etl.html)** is about using LLMs for pipelines, specifically in characterization attempts

## Side Quests

Three side quests in particular this year took a lot of time
1. Infrastructure
2. Training "Foundation Models"
3. Information Bottlenecks

### Infrastructure

I’ll be writing a dedicated post for this within a month or two, but to be able to conduct my personal research and related work, compute is required. I decided against using cloud compute for a number of reasons
primarily privacy, cost control, and the desire to fully understand the systems I depend on.

The homelab that’s been built out primarily serves four needs:

1. Storage
2. Training compute
3. Inference compute
4. Application hosting

The design, implementation, and ongoing maintenance added significant overhead. Every decision, whether it be networking, GPUs, disk layout, OS, etc.. had a learning curve associated with it.

That said, having full-stack control over experiments has opened up doors. Questions that would have felt too expensive or annoying to explore on rented compute are now merely time-boxed. It also
allows for the us of apps that constantly run (e.g., pipelines, agents, etc..).

### Training "Foundation Models"


A lot of claims about model performance, especially things like "domain-adaptation" or "emergent-behavior"
rely on a combination of unknown training procedures, training data, and vague definitions of goals.
I suspect that to work on characterization and domain questions custom trained models will be required.
 Specifically fine-grained control over the provenance of data is needed to make claims about things like
 domain adaptation.
This sounds obvious, but it rules out most off-the-shelf options.
Most (recent) models where the training dataset is known are decoders, not necessarily encoders.

Much of this work didn’t produce blog posts. It involved 
1. building a light, Pytorch based, "framework" for encoder-based models
2. building out an evaluation dataset 
3. building a training machine
4. working on deployment pipelines 


### Information Bottlenecks

I also hit a wall in how I search for papers and organize the information gathered from them. I had a growing list of questions that likely have been partially answered somewhere, 
yet finding relevant work was very difficult.

Current search tools for papers like Google, semantic scholar, or connected papers aren't really cutting it to find relevant papers.
Keyword search fails when terminology drifts. Citation chasing biases towards specific "lines" of research and may not find all relevant work. 
Moreover once papers have been identified, they still need to be organized.

Currently I use a combination of common search tools, manage papers found in a excel, and my conclusions for question in an excel.
This works, but its requires a lot of effort and frankly can be frustrating. I've still found missed papers due to the search tools,
sometimes make copy and paste errors, and none of this can be easily integrated into other tools.
I'm currently working on a custom search tool to help with these questions.


## Goals for 2026

If 2025 was about building scaffolding, 2026 is about climbing it. That being said there is still some work to do.

### Prerequisites

Before any serious research push, a few foundational pieces need to be in place:

1. **Final infrastructure improvements**, particularly around speeding up model hosting and inference. The current setup works, but it leaves performance on the table and breaks from time to time
2. **A research management tool** that supports question-driven exploration rather than paper-driven accumulation.
3. **Finish my light framework for training a foundation model**, even if small, with fully controlled data provenance to serve as a testbed for domain experiments.

### Research

Ultimately, the goal of this setup is to get to clearer research answers, especially around questions that are often hand-waved. For example:
* A rigorous definition of *domain*. I'm currently writing a short opinion paper on this topic
* Predicting downstream performance *before* training, ideally just using downstream task distribution and being able to predict likely performance
* Understanding the carrying capacity of models, i.e., how many dimensions does it take represent a given amount of text