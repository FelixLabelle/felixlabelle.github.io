# Temperature, Tokens, and Long Tales/Tails

While my research interests primarily revolve around characterization (and classification as a result),
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

To evaluate whether model behavior is correlated with temperature, a subset of [euclaise/writingprompts](https://huggingface.co/datasets/euclaise/WritingPromptsX) dataset was used.
A total of 50 prompts about short stories were randomly sampled and used across multiple model, seed, and temperature configurations.
The model was asked to write a short story based on that prompt with the following "system prompt":

```
    system_prompt = (
        "You are a creative writing assistant. "
        "Write a story using the following prompt."
    )
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
```

The models are run using LM studio and exposed using an HTTP server. This choice was made out of expedience and to allow for use of an OpenAI
completions interface, which is relatively standard as of writing. It is worth noting that [it appears possible
to get deterministic results if a single slot is used when using llama.cpp](https://github.com/ggml-org/llama.cpp/issues/7381). Under the hood LM studio uses llama.cpp,hence the use of random seeds.

Each response was generated with 20 different random seeds and 11 temperature settings, uniformly spaced between 0.0 and 2.0.
The random seeds are meant to better understand how variable the paths generated are for a given temperature.
Generation length was capped at 2048 tokens, which is above the max length generated. Sampling parameters were not set.
Three instruction-tuned models were used:
llama-3.2-1b-instruct, llama-3.2-3b-instruct, and meta-llama-3-8b-instruct.

The code used to run the experiments [can be found here](https://github.com/FelixLabelle/temperature_generation_length_experiments).
This setup yielded approximately 33,000 generations in total. Note that
results with temperature 0 were only run once, so there are less 33,000 results.

## Results and Analysis

To better understand whether or not there is a relationship between
temperature generation two approaches are used
1. Summary statistics
2. Correlation

Results were aggregated by averaging generation length across seed–temperature combinations to examine trends across models.
Primarily two statistics were looked at
1. Tokens generated
2. Delta tokens generated 

### Summary Statistics

To examine how sampling temperature affects generation behavior, outputs were grouped and averaged by temperature across all models and seeds.

Mean generation length showed a gradual decline as temperature increased (Table 1). The average length decreased from roughly 536 tokens at temperature 0.0 to 523 tokens at temperature 2.0, indicating a modest contraction in output length at higher sampling temperatures.
The thing to note is that this table doesn't account for variability across different models.

|   temperature |   Mean (tokens) |   Std (tokens) |
|--------------:|----------------:|---------------:|
|           0   |         535.939 |        274.654 |
|           0.2 |         536.293 |        272.566 |
|           0.4 |         540.611 |        271.041 |
|           0.6 |         537.607 |        265.253 |
|           0.8 |         544.993 |        263.526 |
|           1   |         541.177 |        262.383 |
|           1.2 |         538.096 |        258.865 |
|           1.4 |         536.657 |        258.369 |
|           1.6 |         530.935 |        255.534 |
|           1.8 |         523.631 |        252.78  |
|           2   |         523.605 |        253.283 |

The delta metric does account for that. The delta is the difference between the number of tokens generated between temperature 0 and temperature t for a given model M.
The idea is that this helps isolate the change due to temperature and doesn't conflate other factors like the model (like the table above).
Differences in mean token count (Δ tokens) exhibit the same overall trend (Table 2), confirming that higher temperatures generally yield shorter generations. Variability in length, however, increased slightly with temperature, suggesting that while typical completions get shorter, their range widens.
This difference is because the delta accounts for the model used to generate an output.

|   temperature |     Mean Δ |    Std Δ |
|--------------:|-----------:|---------:|
|           0   |  -1.86786  |  87.8109 |
|           0.2 |  -2.27367  | 144.31   |
|           0.4 |   2.04433  | 153.505  |
|           0.6 |  -0.96     | 162.326  |
|           0.8 |   6.42667  | 176.19   |
|           1   |   2.61033  | 187.158  |
|           1.2 |  -0.470667 | 191.741  |
|           1.4 |  -1.91     | 195.618  |
|           1.6 |  -7.63133  | 196.868  |
|           1.8 | -14.9353   | 203.902  |
|           2   | -14.9617   | 211.485  |



### Correlation

Correlation analysis over the summary statistics seems mixed.

Using spearman rank correlation over the delta averages and standard deviation values, temperature was negatively correlated with both mean length (ρ = –0.53) and standard deviation (ρ = 1.00). This indicates that as temperature increases, average completion length decreases while variability increases.
I chose to use summary statistics instead of the raw data (as is typically done), because I wanted to understand how variability increased with temperature
and wasn't sure how to do that using the raw data.

However when measuring the correlation between delta and temperature directly, the correlation is very weak (ρ = -0.03). It's unclear to me why there is such a discrepancy. My best guess is that the average obscures some information such as the model which may skew results.

## Caveats

These experiments are subject to several limitations.

1. all results assume random sampling of prompts and rely on fixed sampling parameters—top-p and top-k values were held constant throughout. No tuning was performed to optimize these settings, which may affect comparability across temperatures or models
2. the analysis was conducted on a single task (story generation) using a single system prompt. Task or prompt specific effects could lead to different trends in other contexts
3. only a limited set of models was evaluated, all from the LLaMA-3 family. Model-specific factors such as architecture (e.g., moe, dense, attention types), size (all the models used are sub 8b), model family (e.g., Qwen, OSS), or post-training objectives could produce different outcomes
4. the experiment script crashed 3 times. As a consequence of that, the results from each run were stitched together manually. I see the expected number of results and have inspected for obvious errors (such as duplicates), but it is worth noting this happened and that this manipulation may have introduced errors
5. the study did not assess generation quality or task performance. The observed differences in length and variability therefore reflect behavioral rather than qualitative effects. If temperature is 
an important factor to quality, it will likely outweigh other practical considerations like time or cost
6. I'm not a statistician and have doubts about the validity of the correlation analysis. I'm currently reading on how to best do an analysis of this type and would apply those techniques in the future
7. The models refuse to generate outputs for certain prompts under certain conditions. It happened only over a couple of the prompts, but I don't think could have skewed results

## Conclusion

From an engineering standpoint, these results seems to indicate that temperature effects practical factors such as inference time and cost when having smaller Llama 3 models tell short stories.
The observed reduction in average generation length at higher temperatures suggests potential efficiency gains, though the accompanying increase in variability may complicate latency estimates in production settings.

More broadly, decoding hyperparameters can influence non-performance metrics in ways that are often overlooked. Estimating their impact on runtime and resource usage could be valuable for teams deploying large-scale generative systems.
Their impact on other aspects like the variability of quality for subjective tasks may also be worth investigating.

To know if this holds, a larger scale study would need to be done addressing the caveats outlined above.
In the meantime, given the limited scope of these experiments, practitioners should replicate similar analyses on their own data and workloads to assess how temperature and related settings affect their specific use cases. 
