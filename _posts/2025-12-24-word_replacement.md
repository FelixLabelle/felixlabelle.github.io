# WIP: LLMs for Text Normalization (i.e., Domain Adaptation)

Work on mapping (link definition). We've deployed mapping models across multiple areas of regulation and clients. When we deploy at new combinations,
there is a drop in performance.

Suspect this is due to shifts in language. This can broadly be viewed as domain shift, although there are underlying mechanisms for performance
drop that are different. In mapping tasks, when working on new domains often there is new language. Language might be rare, unseen, or unknowable without additional
information. Examples include word choice changes, acronyms.

One approach for domain adaptation is making models more robust (list methods), but another approach would be to 
transform the input. Normalization (have reference).

## Description of approach

Approach is simple pipeline that replaces "rare" words with new ones. This is naive, but should work.

Tokenize text with a simple tokenizer

Use an approach to identify rare/unknown words in new Domain

Use LLMs to convert rare words into output

Replace words with relevant information

### Components

This is done using a pipeline. It is 
1. tokenizer
2. Select word to replace
3. Context generation
4. Insertion

## Proposed Mechanisms (Hypotheses)

### Distributional shift

Word insertion/replacement would make changed version to 
be more similar to the original training distribution(s). A simple way of doing 
this would be substitution using a dictionary. 

https://arxiv.org/pdf/2210.03050

### Injecting Knowledge

Acronym resolution or other words not seen during either pretraining or post training

### Compressing Input text

Some words are not well compressed

## Potential Metrics

The idea of these metrics is as diagnostic tools to understand what contributes most to gain

### Domain Similarity to Original Corpus

While working on this it dawned on me that I didn't have a very rigorous definition of domain and domain shift.
There are good references like this paper on domain shift, but in practice for text what constitutes domain shift
is pretty limited. Often in papers domain is defined as "from a different dataset". This handwavy approach 
highlights a need for a more rigorous definition something I will tackle in a future blog post, potentially even an opinion paper

### Validity of LLM Replacements

The model used to generated llm replacements is important. There are a couple Replacements

#### Types of Replacements

While surveying the data I found a couple sources of rare words
1. Errors
	1. Spelling mistakes
	2. Combined words
2. Acronyms
3. Rare words

Measuring the impact of each and relative improvement from each category.
Akin to finegrained performance evaluation (include link to your past post)

#### Ambiguity

The replacements themselves can be ambiguous. Some acronyms can have different meanings across sources or even contexts within a source,
for example API. There is the computer engineering usage, 

Measure the impact of removing ambiguous replacements. Having the model predict these or maybe using a heuristic 
like n-gram similarity to find words that occur in different contexts.

#### Subjective Judgment

Look at the replacements and hand evaluate them  

### Quantify Scale and Type of Replacements

Measuring the correlation between changed vs unchanged documents

Measuring impact of number of changes 

### Measure Rate of Compression


#### Per Replacement

On average per word, is there a saving

#### Per example

For each example that had changes see if there is a correlation between changed in size and performance

Given that documents may have multiple changes there are confounding variables. Maybe be worth just looking at global changes.


## Preliminary Results

Over proprietary, so can't shared. Data likely doesn't reflect most use cases, so your mileage will vary.
Trained on mixed "domains", evaluated on held out "domains"

3.5% of words are replaced

Leads to 1-2% improvement on domain specific models

Further improvements 2-4% when training on entirely held out data

Interestingly commercial models don't improve

Need to do more finegrained analysis

## Next Steps

Measure importance of different changes on output
	Requires more rigorous definition of domain similarity
See if something in particular helps
Once source of improvement is identified maybe change method to further improve that
Replicate work over existing