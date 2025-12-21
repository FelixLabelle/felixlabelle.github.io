# WIP: LLMs for Text Normalization (i.e., Domain Adaptation)

A perennial task at work is [mapping](https://felixlabelle.github.io/2023/12/18/text-similarity-tasks.html#mapping).
We've used mapping models across multiple areas of regulation and clients.
When models are deployed to a new combination of regulation, client or both there is a drop in performance.
There are ways of over coming this like additional training, but ideally models would generalize regardless of the source of the text.
While there may be multiple causes for the drop, like slight differences in definitions of what is considered a good match is across clients (i.e., p(y|x)),
IMO this likely isn't the primary culprit.
Transfers between clients or areas of regulation often result in new language that the underlying model has limited or no exposure to.
Examples include word choice changes or acronyms that only make sense given external context. These issues are common in regulatory text and client-specific documents.
Even for mappings with the same annotators, there can be drops for new "areas" (i.e., its a safe assumption p(y|x) is the same, p(x) is the only change.

One approach to solve this would be model focused (e.g., domain adaptation). However this blog post will be focused on a data centric approach, 
specifically normalization.

## Description of approach

While doing error analysis on mapping examples, the number of acronyms, spelling errors, and offhanded words in different languages present in the data stood out.
A while ago I attended a colloquium where one of the authors of [DictBERT](https://arxiv.org/abs/2208.00635) presented their work. Assuming 
that rare or otherwise unknown words are a cause of performance drops,
a word replacement approach like DictBERT seemed like a viable approach especially on the smaller encoder models we typically use for mapping.

Success is measured by the ability to improve generalization to new "domains" (as defined by area of regulation and client).

A replacement based normalization approach provides a handful of advantages:
1. it can be modified for new clients
2. runs quickly
3. "interpretable" (to some degree, we monitor the replacements and measure their effect on certain key metrics)

There are some downsides, primarily
1. cost of annotation. Unlike the original work we can't necessarily use a dictionary. Often acronyms are domain specific and require annotation
2. replacements are not dynamic (the best replacement might change contextually)
3. it isn't entirely clear what words should be replaced (DictBERT uses a heuristic to select words to be replaced)

Issue 2 is a fundamental limitation of a replacement based approach, but there are ways to mitigate this risk.
Issue 1 can be addressed using models, prompting specifically.

### Implementation

Text normalization is achieved using a pipeline. It is modular so that individual pieces can be swapped out or evaluated independently.
There are two phases, training and inference.

#### Architecture

There are three primary components to the pipeline:
 
1. Tokenizer. It produces word-like tokens and basic offsets used later for insertion. It's a simple regex that returns the word and word position in indices
2. Replacement dictionary. It contains replacements and additional metadata about them (e.g., ambiguity)
3. Downstream model which uses the replaced text (typically an encoder like BERT)

#### Training 
1. Tokenization
2. Create Metadata for labeling. Store context in which tokens are found. Keep surrounding W words in either direction (currently set to 10, based on subjective judgment).
2. Select word to replace. Applies heuristics or thresholds to decide which tokens are candidates for normalization. Currently just using tf/df thresholds. Remove words that occur in more than 50% of documents and account for more than 1% of tokens.
3. Replacement generation.Provide up to 20 contexts (windows of size W stored earlier) and have the model predict different fields
	1. replacement type (more on this below)
	2. ambiguity
	3. replacement
	4. a justification for the values above
4. Post processing of generation. Checking that replacements don't themselves have rare words in them (recursion)

<pre class="mermaid">
flowchart TD
    %% Training pipeline
    T1[Raw Text] -->|raw string| Tok[Tokenizer]
    Tok -->|tokens + char positions| Meta[Create Metadata]
    Meta -->|token list, surrounding W-word windows| Sel["Word Selection (tf/df filter)"]
    Sel -->|candidate tokens + their windows| Gen[Replacement Generation]
    Gen -->|up to 20 W-word windows → model predicts: type, ambiguity, replacement, justification| Post[Post‑processing]
    Post -->|filtered replacements| Ins[Insertion]
    Ins -->|original text with replacements applied| Out[Output Text]

    classDef stage fill:#f9f9f9,stroke:#333,stroke-width:1.5px;
    class Tok,Meta,Sel,Gen,Post,Ins,Out stage;
</pre>
 

#### Inference

1. Tokenization
2. Replacement. Replaces the original token with the generated output while preserving the rest of the text. There may be a better way, but 
the replacements are done backwards, inserting the replacement spans in the text.
3. Pass the text to the downstream task

<pre class="mermaid">
flowchart TD
    %% Inference pipeline
    I1[Raw Text] -->|raw string| ITok[Tokenizer]
    ITok -->|tokens + char positions| IRep[Replacement Engine]
    IRep -->|original text with replacements applied| IOut[Text for Downstream Task]

    classDef stage fill:#f9f9f9,stroke:#333,stroke-width:1.5px;
    class ITok,IRep,IOut stage;
</pre>

## Preliminary Results

The results are over proprietary data, so the underlying data can't be shared. At a high-level it appears to work,
but the effect is very weak. Multiple splits are used to better understand the effects of the data. 

The method was used in two settings
1. held out splits (4) with different sets with models trained on unexpanded data
2. held out splits (4) with different sets with models trained on expanded data

For the word replacement only words that occur in more than 2 documents and more than 5 times are considered.
About 22k rare words were found. ChatGPT 4o was used to generate the word replacements.  About ~1/3 didn't have any replacements so approximately 3.5% of words were replaced on the held out splits. Based on a subjective analysis 
this was due to ambiguous examples or examples without enough examples to determine their meaning. 

The downstream mapping model is a bert-based architecture, with a fair number of variants.

In setting 1 the replacement leads to 1–2% improvement on domain-specific models. Interestingly commercial models don't improve. Need to do more fine-grained analysis to understand why.

In setting 2 further improvements of 2–4% are had.


### Caveats

Bluntly the data I work with is weird and I highly doubt that these gains are universal. The task is rather different than most IR or even similarity tasks.
Moreover the data often has a fair number of rare acronyms. Your mileage will vary and currently I have no method of knowing when this approach is worthwhile
for your datasets.

The pipeline needs to be monitored. Although not mentioned, statistics on the number of ambiguous outputs are tracked. Moreover ambiguous 
or replacements that aren't predicted are not used.

## Suspected Mechanisms (Hypotheses as to Why this Works)

I think there are multiple reasons replacement should/could work. I'm positing them here so future experiments can be focused
on determining why and when normalization through word replacement might work. In specific there are 3 causes I suspect.

### Distributional shift
<!-- TODO: Review from here on out -->
Word insertion/replacement might reshape the text to have a distribution more similar to the original pre-training or post-training distribution(s) for the downstream model.
An argument could be made that implicitly this happens since the model is likely going to favor web like words and phases in the replacement models pretraining corpus (being a language model), but I have no proof for this.
The only counter argument I see is that the model being used to replace words may not be the same one doing the underlying task. GPT-4o and BERT have overlap in terms of training data,
but 4o has a much larger corpus and likely a different distribution.

Measuring the shift created by word replacement requires a definition of domain. For our purposes we'll use token distribution, since this makes it easier to shape the output. For open-source models with training data available we can measure token distrbution
and use logit biases to make replacements more similar to the pretraining corpus. Similarity to pretrain data has been shown to [improve performance across a number of tasks](https://arxiv.org/abs/2507.12466). This previous
work is in decoder not encoders we might use in downstream applications.
The corpus could also be made more similar to finetuning data. In general shaping the replacements to be similar can be achieved through logit biasing.

To measure the effect that the change in distribution has I would propose measuring the change in similarity to both the pre-training and post training distribution and how that correlates to 
performance. In other words, do replacements which make a given item more or less similar to the pretraining distribution improve performance?
Both these actions can be framed as binary;
1) whether or not a text is closer to a given distribution after replacement
2) whether or not the mapping metric (lets say NDCG) is doing better

Framing the results as binary is naive, but would allow for use of a simple statistical test. The Phi coeffecient would work well for this.
This assumes that the representation results in some number of more distant. In all likelihood it might make more sense to use some statistical
measure of corellation between the document and the distributions and performance.

### Injecting Knowledge

Beyond the distribution of words, sometimes knowledge will be missing. Acronyms or other words rarely or not seen during either pretraining or post-training can be expanded into more common forms.
This effectively injects external knowledge that the base model may not reliably infer on its own.
The reason this would be effective is that rare or domain specific words will likely have the most "information". This is the principle on which
tf-idf works. If we are missing critical/rare words we can't effectively do the task as critical words are missed. For example,
if a document is referring to a region, but an acronym is used the mapping will be missing (e.g, APAC vs Asia Pacific).
While surveying the data I found a couple sources of rare words. These categories are not exhaustive, but they cover most observed cases.

1. Errors  
	1. Spelling mistakes  
	2. Combined words
2. Acronyms  
3. Rare words
4. Other
	1. Translations

The first questions would be the impact of each and relative improvement from each category. Potentially some types of information
are more critical/useful? A simple analysis looking at # of changes in a given type of change and performance could make sense in this context.

### More Efficient Representations of Input text

Some words are not well tokenized. Replacing improperly tokenized with more common equivalents may reduce representational inefficiency while giving more meaningful tokens.
I do think, subjectively, that the replacement tokens are more meaningful.

```
APAC
[AP][AC]

Asia Pacific
[Asia][ Pacific]
```




There are counter examples where less tokens are created originally, see the examples below: 
```
AML
[AML]

Anti Money Laundering
[Anti][ Money][ Laundering]
```

```
FATF
[F][AT][F]

Financial Action Task Force
[Financial][ Action][ Task][ Force]
```


Regardless the tokens do appear meaningful. I think there are two separate claims here that need to be measured.
The first is the correlation between changes in size and performance. Given that documents may have multiple changes there are confounding variables, so it may be worth just looking at global changes.
To measure correlation we'll again frame this problem as binary. Texts that are the same or smaller vs longer and see if there is a correlation.

For the meaningfulness of the tokens, this is a bit trickier to measure and define. My only thought here is to measure how many subwords are used
(i.e., the fertility of the tokenizer). For now, I'd focus on the first.


## Next Steps

The next steps are to measure the importance of different changes on output. 
Moreover it may be important to replicate work over existing benchmarks and non-proprietary datasets
and see if the results hold.

<!--
#### Types of Replacements



What is the % of documents changed. % of words changed

#### Ambiguity

The replacements themselves can be ambiguous. Some acronyms can have different meanings across sources or even contexts within a source, for example API. There is the computer engineering usage, among others. Measure the impact of removing ambiguous replacements, either by having the model predict ambiguity or using a heuristic like n-gram similarity to find words that occur in different contexts.

Currently model predicts whether or not ambigious. Haven't validated performance of that.
-->