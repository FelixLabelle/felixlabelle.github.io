# Original Work: Pilot on when to Finetune vs Pilot

An engineer needs to understand the tools at his disposal and their pros and cons 
to use them appropriately. When it comes to NLP, sometimes it's easier to decompose a problem into classification 
tasks or add classification to make decisioning and reducce costs.
Now a days classification can be done in a variety of different ways.

We've all seen vague post non-sense about finetuning, rag etc improving models.
Lol I'm a bit late to working on this, since after looking up the initial tweets I remember
were from 2023
https://x.com/karpathy/status/1655994367033884672

Here's a reminder of what those charts look like,
![Research hub landing page](/images/vague_post_non_sense.png)
![Research hub landing page](/images/vague_post_non_sense_2.png)
Half the time the margins are made up/pulled straight out of the posters ass.

I want to understand how true these statements are. Decided to do a small
scale pilot and see if these claims bear any truth.

## Research Questions

1. How do prompt-based methods perform on a swath of classification tasks?
	1. Variability of prompt-based approach predictions (models are non-deterministic)
	2. Effects of aggregation methods on performance
	3. Effect of k-shot as k -> inf
	4. Effect of RAG on classification
2. How do prompt-based methods compare to finetuning on smaller (Encoder) models
	1. Data efficiency (when do models 
	2. Effects of model size on finetuning

## Methodology

Use classification over single texts. Easier to evaluate
and control for variables to start
### Datasets

Use prompt-source, because
1. Has multiple prompts for giving classification datasets
2. Large amount of datsets (100+)
3. Variable size

### Evaluation


## Results


## Caveats

## Next Steps

1. Extend to multi-text tasks (e.g., NLI)
2. Extend to other classification tasks (e.g., span extraction)
3. Account for contamination

## Closing Remarks