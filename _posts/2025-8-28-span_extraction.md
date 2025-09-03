# WIP: Using Landmarks to Extract Spans with Prompting

Over the last couple of years prompting and more generally text generation have become more common place in NLP applications. Instead of 
bespoke tools that require training data, the default is to use prompts and generate text. This approach to NLP does present some advantages,
such as tackling more complex use cases and  ease of development. 
However the open-endedness provided by text generation is not always necessary and can even be detrimental for certain applications.
Reasons for this in my opinion are:
1. Open-endedness can make evaluation harder (i.e., less or no automated metrics available)
2. Cost of generating tokens and sizes of models needed for more complex tasks

Sometimes tasks can be solved with more closed off solutions.
Often when we get asks to "summarize" documents, the underlying ask is not a summary, but rather 
to highlight use specific information such as legal obligations.
For that reason we sometimes opt to 
frame the task as a span extraction task rather than a generative one. Summarization can be applied after the fact if conciseness is desired,
 but based on attributable spans.
This makes verification easier and makes picking metrics feasible. Extracting relevant spans can be done with existing models,


A limitation to span extraction is that we may not always have access to training data
to pull relevant spans or pretrained models with reliable enough performance. In this case using prompting
and the benefits of a larger model may be interesting. What if we can have our cake and eat it too?

We can prompt a model to pull spans verbatim from the text, but this risks:

1. Hallucinations (the funniest for me is when models correct spelling mistakes in the original text)
2. Have a higher cost since we are generating existing tokens

Rather than pull out entire spans, an alternative is to create landmarks that anchor spans and span specific information.
Rather than generate entire spans, we can pull out one or more words at the beginning and end of a span
to anchor it. This reduces the amount of tokens while having a calculable way of balancing the number of tokens required.


## Landmark Method

We can prompt a model to identify the **start** and **end anchors** of relevant spans.
Anchors can be a fixed number of words (e.g. 1–5 tokens) rather than the entire span.

A simple prompt template might look like this:

```
input_text = "All firms must report their financial holdings quarterly to the regulator. Smaller firms are exempt from annual disclosures." 
prompt = """You are an annotator. Given the text below, find the span(s) that mention regulatory obligations.  
For each span, return the first 3 words and the last 3 words verbatim from the text.  

Text:  
 

Output format:  
{"start_anchor": "...", "end_anchor": "..."}
"""
```

Example output:

```
span_anchors = [
  {
    "start_anchor": "All firms",
    "end_anchor": "the regulator."
  },
  {
    "start_anchor": "Smaller firms",
    "end_anchor": "annual disclosures."
  }
]
```

This way, instead of regenerating the entire span, we only generate a handful of tokens.
The original span can then be reconstructed by searching for the anchors inside the text.
This could be done using a regex:

```
import re

anchors_texts = []
for span_anchor in span_anchors:
    start = re.escape(span_anchor["start_anchor"])
    end = re.escape(span_anchor["end_anchor"])
    
    pattern = rf"{start}.*?{end}"
    match = re.search(pattern, input_text, flags=re.DOTALL)
    
    if match:
        anchors_texts.append(match.group(0))
	else:
		print(f"Invalid span found {pattern}")

print(anchors_texts)
```

## Metrics

There are two types of metrics:

1. **Task-specific metrics**
2. **Anchor quality metrics**

**Task-specific metrics** could be any traditional span extraction evaluation. These include exact match or token-level overlap with ground truth spans. Think ROUGE, BLEU. 


**Anchor quality metrics** focus on how well the anchors serve their purpose:
* Ambiguity rate: how often an anchor occurs more than once in the document. This can be minimized with longer anchors, but comes at a cost
* Anchor placement: whether anchors actually map back to the intended span
* Anchor failure rate: Percentage of the time anchors cannot be found in the text

## Applications and Caveats

Caveats:
* Anchors are brittle when text is heavily paraphrased.
* If the model outputs incorrect anchors,
* Search-and-reconstruction assumes the input text is available and relatively clean (e.g., no OCR errors or tokenization mismatches).
* this approach only makes sense in cases where spans are significantly longer than anchors

Viable use cases, in my experience, are:
1. Regulatory obligation extraction. Obligations can be rather longer, often 1-3 sentences
2. List parsing (e.g., pulling out enumerated conditions or clauses)




## Next Steps

Immediate next steps would be:

1. **Benchmarking**: Test over a publicly available span extraction dataset (e.g., SQuAD, contract clause extraction).
2. **Comparisons**: Measure against two baselines:

   * A pure prompt-based extractor that generates spans directly.
   * A smaller fine-tuned model trained for extraction.
3. **Trade-off analysis**: Evaluate anchor length vs. accuracy vs. cost.

Longer-term, this method could be explored as a hybrid: use anchors for **cheap weak labels** at scale, then bootstrap a smaller, cheaper model that can handle direct span extraction. That gives you the benefits of large models without being locked into them at inference time.

## Closing Remarks

I expect to see more methods around prompting and LLMs that reduce the open-endedness of generation. 
This is already occurring in the form of guided generation or other methods.
I suspect this is will be due to; diminishing returns for larger models and training runs, the cost (primarily environmental) of running larger models,
and regulatory requirements that require accountability when it comes to measuring performance (at least for certain use cases).

One task that I haven't seen this trend is span extraction and I think anchors may serve this niche well. 
This is still an experiment, and I don’t know yet how well anchors will hold up across tasks. Anchors likely aren’t a silver bullet, they inherently trade-off between cost and precision.
Even if this approach doesn’t replace other methods, it can be a useful tool in the toolbox.