# Improving the NLP Tool Kit: Characterization

The last couple of years have seen a shift in NLP research and products, moving from solving specific problems to more general solutions.
Five or six years ago, papers often proposed a single dataset or architecture aimed at specific tasks.
While some early work on better representations that could improve performance across multiple tasks did exist like
[word](https://arxiv.org/abs/1301.3781) [embeddings](https://arxiv.org/abs/1607.01759), [contextual embeddings](https://aclanthology.org/N18-1202.pdf),
or [BERT CLS embeddings](https://arxiv.org/abs/1810.04805). It is worth noting most of the datasets used to evaluate how well representations generalized were smaller
in scope like [GLUE](https://arxiv.org/abs/1804.07461).
Similarly, the few successful NLP tools in deployment at the time had a single function, such as [search](google.com) and [translation](https://translate.google.com/) which
required quite a bit of work (engineering, data, hacks) to get working properly.

With the [rise of GPT models](https://arxiv.org/abs/2203.02155) gone are the days of gathering data and training a specialized model for some scoped down task.
Models can now [generate open ended outputs and therefore answer and be evaluated on a large number of tasks](https://arxiv.org/abs/2005.14165).
Accordingly the expectations of what problems models should be able to solve has changed as well.
Implicitly tools are increasingly expected to excel at something beyond a single purpose; terms like "AGI" are [thrown](https://arxiv.org/abs/2303.12712) [around](https://openai.com/charter/).

While these advances are interesting, sometimes it feels as though a step was skipped. The field moved directly from solving individual problems to attempting to solve all problems.
There are two reasons for this
1) even these improved models don't get perfect results
2) as an engineer there is still a need for task specific solutions (e.g., performance, cost, run time, etc.).

When scoping out specialized solutions such as these, we need the ability to estimate resource requirements and confirm solution viability before implementation. 
In spite of recent advances, IMO, there is still a lack of tools to help make design decisions for 
specific tasks. There's no straightforward way to build a maintainable, debuggable model-based classifiers quickly and efficiently or predict how a solution will perform in the field.
Instead of striving to solve problems "in general," we should aim to solve general problems about ML effectively. Understanding how models perform in certain settings is
just as interesting if not more so than a model that can do everything in my opinion. Characterization is one way of achieving that.

## Overarching Goal: Characterization

One way to scope projects is using measures that are predictive of downstream performance/behavior.
This is analogous to **characterization** in fields like material sciences; measures that help determine a tool's technical capabilities. 
While this is not necessarily a novel idea even for NLP, as some instances already exist such 
as scaling laws, the work I'm aware of is disparate and characterization is not a field in its own right.

### Examples of Characterization in Other Engineering Fields
Characterization is used in other engineering fields. As someone who is more familiar with electrical engineering, I will primarily give examples from that field. 

In circuit design, characterization involves using measurable properties like resistance, capacitance, and inductance to predict how components will function in a circuit. These properties allow engineers to:  
- Design filters or amplifiers without needing to build physical prototypes.
- Understand voltages and currents at different points in a circuit, for example this can help calculate gains on an op amp (using a voltage divider).
- Predict noise within a circuit or potential areas where issues could arise.

Similar examples exist in other engineering disciplines like civil or mechanical engineering. Two measures that come to mind are tensile strength, elasticity. These can help select the correct material for building 
a given project. 

These examples highlight how measurable properties can save time and resources by providing actionable insights during the design phase,
ensuring that a solution will meet desired outcomes without excessive trial and error.

### Characterization in NLP

We want to turn a given approach (e.g., an embedding) into a bunch of numbers and assess viability for a given downstream task (e.g., detecting the emotion within a document).
To simplify this task I want to start by focusing on classification tasks using a linear layer over a single embedding. This specific approach was chosen because:
1. classification is still a very common task.
2. finetuning over an embedding for classification is common. Even models being trained in their entirety can be analyzed by just examining the last layer.

With this framing in mind, the first step is to identify questions we might want to answer with measures:
1.  Which [embedding space](https://en.wikipedia.org/wiki/Latent_space) would be best for this task?
2. [domain invariance](https://arxiv.org/abs/2102.05082) within the target downstream task.
3. How data length impacts performance for a proposed task.
4. Do I have adequate coverage of examples? What type of data should I be labeling? Do I have enough data to train a model?

Second step is to find measures that are useful in answering these questions and validate them (see the road map below).
Once these measures are validated rather than try a few attempts based on intuition we could down select using appropriate measures.
Moreover, these measures could be used to create better representations for a given problem. Let's say we know a given embedding can only represent 
documents of length 512 for [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval), we could do something like fuse multiple embeddings from different models or layers.

#### Assumptions

This vein of work was chosen with several assumptions in mind. These are axiomatic and admittedly potentially flawed.

1. **Made-to-specification solutions are more efficient (resource-, time-, and cost-wise) than open-ended approaches for well-defined problems.** This doesn’t mean prompting or [other zero-shot approaches](https://huggingface.co/tasks/zero-shot-classification) to a specific task are bad, but specialized systems can simplify assumptions to reduce costs or even improve performance.
2. **Most engineering problems can be well-defined.** This is based on my intuition and observations at work.
3. **Complicated problems can often be broken down into multiple steps.**
4. **Classification is a good focus for characterization:**
   - It is easier to evaluate. Open-ended generation has a vast space of valid outputs, making it inherently harder to evaluate. Even within that space, evaluations often focus on specific facets that can be reframed as classification tasks (e.g., [Factuality of a summary](https://huggingface.co/datasets/google-research-datasets/xsum_factuality)).
   - It is often sufficient for the task at hand. At we work we often need to determine if documents are well written, which can be broken down into facets. We can use classifiers to either extract facets from text or even just train a classifier to detect whether they are present.
   - It is versatile and can serve as filters, information enhancers, or a tool for finding similar or "confusing" items.
5. **There exists measures that can be used to capture downstream performance across a variety of settings.** That's not to say they will be easy to find, just that I think they do exist.
6. **Being a non (m|b|tr)illion-dollar entity focusing on simpler tasks and characterization is a good way to compete/create useful models.**

### Relevant Work

This is by no means a comprehensive list, if you see a missed reference feel free to comment or contact me.
Part of the difficulty with the task of collecting a list of relevant works is that while it
is possible to find small clusters of related works, but I haven't yet
found a well flushed out field.

This section is likely going to be a living list. I'm not as up to date with all relevant literature as I'd like to be so some sections will be sparse for now.

1. Transfer learning between tasks, representation learning
	1. https://arxiv.org/abs/2005.00628
	2. https://aclanthology.org/P19-1439/ 
	3. https://arxiv.org/pdf/2005.00770.pdf
2. Probing
	1. https://aclanthology.org/2022.emnlp-main.793.pdf 
	2. https://arxiv.org/pdf/2211.06420.pdf 
	3. https://arxiv.org/pdf/2202.12801.pdf
3. Few sample finetuning
	1. https://arxiv.org/pdf/2006.05987.pdf 
4. Finetuning ideas
	1. https://aclanthology.org/2023.findings-acl.889.pdf
5. Predicting performance
	1. https://aclanthology.org/P18-2072/
6. Meta learning literature (honestly I'm less familiar with this area of research. Need to read more here)
7. Optimal Experiment literature (honestly I'm less familiar with this area of research. Need to read more here)
8. [This paper primarily discusses the role of choice of metric and statistics on how strong apparent correlation is between # FLOPs and downstream performance.](https://arxiv.org/pdf/2406.04391). This paper gets rather close to characterization, but stops short of considering measures beyond pretraining FLOPs.
## Road Map for 2025

Below is a roadmap that defines initial steps to develop characterization in NLP. The steps (for now) consist of:

1. **Creating a large classification-only dataset** to enable meaningful experiments. This won't be an effort from scratch, rather aiming to curate existing benchmarks and datasets. I want to do this rather than use a benchmark like GLUE, SuperGLUE or BigBENCH because 1) it allows for a deeper understanding of the data present 2) there are potentially [questions of quality with some existing datasets](https://cs.nyu.edu/~davise/Benchmarks/BIG-bench.html).
2. **Conducting early experiments** to identify measurable properties that correspond to downstream performance, even in limited settings. The first attempted measure will likely be carrying capacity (what length of text a given embedding can capture).
3. **Validating pretraining, finetuning, or other techniques on classification tasks using these measures** to confirm the validity of the identified properties.
4. **Expanding the dataset to other types of tasks** such as information retrieval, sequence tagging.

## Closing Remarks

Characterization is an attempt to make NLP projects more predictable, efficient, and effective.
By focusing on measurable properties and how those translate to technical capabilities, the gap between abstract model performance and real-world engineering needs can be bridged.
The hope is that this groundwork will help create practical tools and frameworks that engineers can rely on when designing NLP systems.


<!--
papers I need to read
https://arxiv.org/abs/1811.04871
https://arxiv.org/pdf/2210.07352
https://arxiv.org/pdf/2201.10474
-->