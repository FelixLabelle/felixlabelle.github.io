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
	1. Variability of prompt-based approach predictions
        1. Variation across different decoding temperatures (models are non-deterministic)
        2. Variation across prompts (brittleness)
        3. Reasoning before vs after classification
    2. Effect of additional systems around a prompt
	    1. Effects of aggregation methods on performances
        2. Effect of k-shot as k -> inf
	    3. Effect of RAG on classification
    3. General study of model based performance
        1. Model family
        2. Model size
        3. Reasoning
        4. Generation
    4. Effect of data on performance
        1. Was data previosly seen by model
        2. Domain variability
2. How do prompt-based methods compare to finetuning on smaller (Encoder) models
	1. Data efficiency
	2. Effects of model size on finetuning
    3. Effect of model family (BERT, ModernBERT, Ettin)
3. Can we predict these trends?

<!-- TODO: specify which this post will tackle -->

## Methodology

Use classification over single texts. Easier to evaluate
and control for variables to start

### Datasets

<!-- Try again with datsets, lets see if we can expand this
Lets control scope through a three phased approach -->
Use prompt-source, because
1. Has multiple prompts for giving classification datasets
2. Large amount of datsets (100+)
3. Variable size

Wound up with 17 datasets

### Evaluation
Use precision, recall, F1

### Models

### Experiments

Over every prompt for every datataset run across every model
run in class before class after setting


Currently working on RQ2, but this will take longer.

Define expectations and how each will be measured in experiment described above
#### RQ 1.1.1
Variation due to radnomness
#### RQ 1.1.2
Variation across prompts

#### RQ 1.2.1

#### RQ 1.2.2

#### RQ 1.3

## Results


## Caveats


## Next Steps

Finish RQ

1. Extend to multi-text tasks (e.g., NLI)
2. Extend to other classification tasks (e.g., span extraction)
3. Account for contamination

## Closing Remarks
