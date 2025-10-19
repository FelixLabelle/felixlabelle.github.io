# Temperature, Tokens, and Long Tales (Tails)

** NOTE: this post is in a draft state, it will be finalized tommorow. Currently checking for formatting issues **

While my research interests primarily revolve around classification,
my day job and the field more broadly (currently) revolve more around text generation. Recently
at work we have deployed models on a platform that has a hard cutoff for run times to avoid
run away processes hogging compute in production.
A generation based tool we've deployed in that environment has been getting timed out due to this practice.
We have been experimenting with optimizations to avoid the time outs, primarily around breaking
generations into multiple steps or reducing the tokens sent in or produced.

Another engineer noticed that lowering the temperature reduced generation times over our 
dev set. This made me wonder, is there a relationship between temperature and generation length.
Two questions in particular came to mind:
1. What would be the mechanism of action for this?
2. Can this be measured empirically beyond our relatively specific and small test set?

## Proposed Mechanism(s) of Action

This post won't really dive into the cause, but rather focus on whether or not such 
a relationship can be observed. That being said, here are two potential explanations I came up with for the initial observation
of higher temperatures corresponding to longer generations:
1. diluting the probabilities (i.e., increasing the temperature) decreases the chance of predicting EOS, which makes sentences longer
2. the paths explored at higher temperatures are on average longer

Both of these causes would likely depend on the underlying model used and potentially the underlying task.
Given the finite of compute available (to me), the following experiments 
will do their best to account for those confounding variables as much as possible.

## Experiment

To evaluate whether model behavior is correlation with temperature, a subset of [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/WritingPromptsX) dataset was used.
A total of 50 prompts about short stories were randomly sampled and used across multiple model, seed, and temperature configurations.

Each prompt was generated with 20 different random seeds and 11 temperature settings, uniformly spaced between 0.0 and 2.0.
The random seeds are meant to better understand how variable the paths generated are for a given temperature.
Generation length was capped at 2048 tokens, which is above the max length generated. Sampling parameters were not set.
Three instruction-tuned models were used:
llama-3.2-1b-instruct, llama-3.2-3b-instruct, and meta-llama-3-8b-instruct.

The code used to run the experiments [can be found here](https://github.com/FelixLabelle/temperature_generation_length_experiments).
This setup yielded approximately 33,000 generations in total. Results were aggregated by averaging generation length across seed–temperature combinations to examine trends across models and sampling conditions.
Primarily two statistics were looked at
1. Tokens generated
2. Delta tokens generated (difference between temperature 0 and temperature t for a given model M)

## Results and Analysis

To better understand whether or not there is a relationship between
temperature generation two approaches are used
1. Summary statistics
2. Correlation
### Summary Statistics

To examine how sampling temperature affects generation behavior, outputs were grouped and averaged by temperature across all models and seeds.

Mean generation length showed a gradual decline as temperature increased (Table 1). The average length decreased from roughly 534 tokens at temperature 0.0 to 517 tokens at temperature 2.0, indicating a modest contraction in output length at higher sampling temperatures.
The thing to note is that this table doesn't account for variability created by different models.

|   temperature |   Mean (tokens) |   Std (tokens) |
|--------------:|----------------:|---------------:|
|           0   |         534.291 |        280.155 |
|           0.2 |         530.63  |        277.942 |
|           0.4 |         535.577 |        277.074 |
|           0.6 |         533.973 |        270.563 |
|           0.8 |         540.675 |        269.373 |
|           1   |         537.099 |        268.28  |
|           1.2 |         533.401 |        263.439 |
|           1.4 |         532.768 |        263.404 |
|           1.6 |         524.866 |        259.892 |
|           1.8 |         518.234 |        257.689 |
|           2   |         517.2   |        257.602 |


Differences in mean token count (Δ tokens) exhibit the same overall trend (Table 2), confirming that higher temperatures generally yield shorter generations. Variability in length, however, increased slightly with temperature, suggesting that while typical completions get shorter, their range widens.
This difference is because the delta accounts for the model used to generate an output.

|   temperature |     Mean Δ |   Std Δ |
|--------------:|-----------:|--------:|
|           0   |   0        |   0     |
|           0.2 |  -3.66028  | 144.545 |
|           0.4 |   1.28617  | 155.605 |
|           0.6 |  -0.31773  | 165.152 |
|           0.8 |   6.3844   | 178.476 |
|           1   |   2.8078   | 191.071 |
|           1.2 |  -0.889362 | 195.845 |
|           1.4 |  -1.46842  | 199.781 |
|           1.6 |  -8.87643  | 200.392 |
|           1.8 | -15.5093   | 209.047 |
|           2   | -16.5425   | 216.161 |



### Correlation

Correlation analysis supported these findings. Using spearman rank correlation over group averages, temperature was negatively correlated with both mean length (ρ = –0.64) and standard deviation (ρ = 1.00). This indicates that as temperature increases, average completion length decreases while variability becomes more pronounced.


## Caveats

These experiments are subject to several limitations.

1. all results assume random sampling of prompts and rely on fixed sampling parameters—top-p and top-k values were held constant throughout. No tuning was performed to optimize these settings, which may affect comparability across temperatures or models.
2. the analysis was conducted on a single task (story generation) using a single prompt template. Task- or prompt-specific effects could lead to different trends in other contexts.
3. only a limited set of models was evaluated, all from the LLaMA-3 family. Model-specific factors such as architecture (e.g., mixture-of-experts vs. dense), size, or post-training objectives could produce different outcomes.
4. the self-hosted LLM failed to generate outputs for about 3,000 runs of the 8B model. I'm currently investigating why. I don't think this effects the overall results, but it is a discrepancy worth noting.
5. the study did not assess generation quality or task performance. The observed differences in length and variability therefore reflect behavioral rather than qualitative effects. If temperature is 
an important factor to quality, then these results will be less interesting.


## Conclusion

From an engineering standpoint, these results seems to indicate that temperature effects practical factors such as inference time and cost.
The observed reduction in average generation length at higher temperatures suggests potential efficiency gains, though the accompanying increase in variability may complicate latency estimates in production settings.

More broadly, decoding hyperparameters can influence non-performance metrics in ways that are often overlooked. Estimating their impact on runtime and resource usage could be valuable for teams deploying large-scale generative systems.

To know if this holds, a larger scale study would need to be done addressing the caveats outlined above.
In the meantime, given the limited scope of these experiments, practitioners should replicate similar analyses on their own data and workloads to assess how temperature and related settings affect their specific use cases. 
