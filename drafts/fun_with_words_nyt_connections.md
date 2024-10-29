# Fun With Words: NYT Connections

NLP isn't all business, it can be fun too. I enjoy word puzzles from time to time, like NYT Connections. It is a game where you try to group 16 words into 4 groups of 4, each with different themes. There are 4 difficulties of themes, easiest -> hardest.

Below is an example:
 
![Start of NYT Connections Game for 10/21/2024](/images/nyt_start_10-21-2024.png)

You are given 4 tries to group the 16 words above. For the example above, here are examples of the groups and themes.

![Completed NYT Connections Game for 10/21/2024](/images/nyt_complete_10-21-2024.png)

I wanted to know if NYT Connections could be solved by machines. This blog is a, somewhat, naive first attempt at solving the game 

## Experiment

I decided to frame this problem as a generative one. There are other framings, see the next steps for those. I chose this framing since it seemed like a strong and
easy to implement baseline. The model is used to generate a structured output that indicates which words are grouped together. The following subsections
explore the setup of the experiments and different design decisions.

The code used to run the experiments can be [found here](https://github.com/FelixLabelle/connections_solver/blob/main/generate_experiments.py).

### Dataset

[This repo by "Eyefyre"](https://github.com/Eyefyre/NYT-Connections-Answers) was used. It automatically updates through Github Actions, which saved me the effort of building 
a pipeline and a way of updating it regularly. Rather than continuously update the dataset, I cloned the repo on the 12th and have not updated it since.

### Design Decisions

#### Differences From the Game

This version is not turn based, rather the model has a single shot to find all 4 groups. While a turn-based approach could be interesting (it enables for use of 
multi-turn strategies and self-correction), I've decided to ignore it for now to simplify things.

#### Sample Size

Some of the larger models, especially if not quantized, don't fit on my GPU and overall runtimes can take up to a day **for a single experiment**.
A sample size of 50 reduced max run times to 3-4 hours which made the experiments run in a reasonable amount of time.

#### Seed

Set fixed seeds for pretty much all random operations such as shuffle and passed in a seed wherever possible. The seed was 42, because 42.

#### Quantization

4 byte quantization was used to fit even the larger models on GPU. The code can be run using other quantizations, but currently that hasn't been done.

#### Output Structure
Use of JSON as the output's structure. I wanted an output
format that was easy to work with and easy to validate using regexes/other programming tools.
This may well effect impact performance in a way that makes comparing models difficult, 
but there likely would be no "fair" format. Here is a function which generates regex describing an example valid output,


```
def generate_validation_regex(words):
    qouted_words = [f"\"{word}\"" for word in words]
    words_regex = f"({'|'.join(qouted_words)})"

    backreference_terms = [""] + ["(?!{})".format("".join(["\\{}".format(j) for j in range(1, i+1)])) for i in range(1, 16)]

    regex_str = r'\[\s*{{\s*"words"\s*:\s*\[\s*{words}\s*,\s*{back1}{words}\s*,\s*{back2}{words}\s*,\s*{back3}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back4}{words}\s*,\s*{back5}{words}\s*,\s*{back6}{words}\s*,\s*{back7}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back8}{words}\s*,\s*{back9}{words}\s*,\s*{back10}{words}\s*,\s*{back11}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*,\s*{{\s*"words"\s*:\s*\[\s*{back12}{words}\s*,\s*{back13}{words}\s*,\s*{back14}{words}\s*,\s*{back15}{words}\s*\]\s*,\s*"theme"\s*:\s*"[^"]*"\s*}}\s*\]'.format(words=words_regex,
        back1=backreference_terms[1], back2=backreference_terms[2], back3=backreference_terms[3],
        back4=backreference_terms[4], back5=backreference_terms[5], back6=backreference_terms[6], back7=backreference_terms[7],
        back8=backreference_terms[8], back9=backreference_terms[9], back10=backreference_terms[10], back11=backreference_terms[11],
        back12=backreference_terms[12], back13=backreference_terms[13], back14=backreference_terms[14], back15=backreference_terms[15]
    )

    return regex_str.strip()
```

The gist of it is that words is generated from a fixed set and words can't repeat (hence the lookarounds, see back1->back15).

#### Guided Generation

For guided generation [Outlines](https://github.com/dottxt-ai/outlines) and a [Pydantic](https://docs.pydantic.dev/latest/) data model.  I couldn't figure out how to add a condition
to Pydantic to limit generation for words to one instances. Consets don't appear supported by Outlines and I kept getting a "can't be hashed" error.

Later I wrote a regex that could do implement word wise uniqueness (see above),
but [learned Outlines doesn't support lookarounds](https://github.com/dottxt-ai/outlines/issues/456). This means even guided outputs can be flawed.
Because of the effort required that late in the project, that is expected. Errors in construction will be accounted for in the metrics.

#### Decoding Settings

Used greedy decoding. Adding in different decoding approaches and additional hyper-parameters could be interesting, but would require more experiments and computations.

#### Prompt

I used a very simple prompt and didn't tweak. While models are brittle, messing around with prompts to improve performance is trivial IMO and falls outstide of the scope of what I wanted to do.
The prompt used was the following:

```
You are an expert problem solver. You are doing the "connections" puzzle. You will receive 16 words.

Find groups of four items that share something in common.

Select four items and tap 'Submit' to check if your guess is correct.

Here is an example with 2 groups of 4 fours words

Your words are: Bass;Opal;Trout;Salmon,Ant,Drill,Island;Flounder

Create a JSON with your reasoning and the groups of four found.

{'groups' : [{'theme' : 'FISH', 'words' : ['Bass', 'Flounder', 'Salmon', 'Trout']},
{'theme' : 'FIRE __', 'words' : ['Ant', 'Drill', 'Island', 'Opal']}]}

Categories will always be more specific than "5-LETTER-WORDS," "NAMES" or "VERBS."

Each puzzle has exactly one solution. Watch out for words that seem to belong to multiple categories!
```

### Hyper-parameters

Below is the list of hyper-parameters that were varied and their values. The experiment runs essentially a grid search of the following hyper-parameters. See [generate_experiments.py]().

#### Models

Tried both llama and the Qwen models. I tried using [Mistral models, but they were not supported by Outlines](https://github.com/dottxt-ai/outlines/issues/1222). I used
```
model_ids = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]
```

I didn't use ChatGPT since it doesn't have open weights and didn't feel like spending the money to try this out. Maybe on another iteration, but for now it's good.

#### K-Shot

Different 
`k_shot_options = [0, 1, 3, 5]`

#### Guided vs Unguided generation

Tried both guided and unguided generation.
`use_structured_prediction_options = [True, False]`

### Metrics

Not turn-based so it excludes certain metrics such as number of turns required to solve, types of errors, model confusion per turn, etc..

A good metric would 
1. Measure how many matches the model got
2. Be able to account for partial success
3. Aware of difficulty of a given group
4. Account for theme accuracy
5. Detect whether or not an output is valid

Also looking at theme appropriateness, would be interesting but a hard task in its own right. A subjective analysis.

For now I used two metrics
1. group-level accuracy, what number of groups the model got correct, because that's easy to implement. The accuracy could further be segmeted based on the difficulty
2. validity of the output using a regex

## Results

### Raw Results
| model_id                              | use_structured_prediction   |   k_shot |   mean_accuracy |   percentage_format_passed |
|:--------------------------------------|:----------------------------|---------:|----------------:|---------------------------:|
| meta-llama/Llama-3.2-3B-Instruct      | False                       |        5 |       0.06      |                       0.58 |
| Qwen/Qwen2.5-3B-Instruct              | True                        |        0 |       0.035     |                       1    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | True                        |        5 |       0.02      |                       1    |
| Qwen/Qwen2.5-1.5B-Instruct            | False                       |        3 |       0         |                       0.38 |
| Qwen/Qwen2.5-3B-Instruct              | False                       |        1 |       0.04      |                       0.84 |
| Qwen/Qwen2.5-3B-Instruct              | False                       |        0 |       0         |                       0    |
| Qwen/Qwen2.5-1.5B-Instruct            | True                        |        0 |       0.02      |                       0.98 |
| Qwen/Qwen2.5-14B-Instruct             | True                        |        3 |       0.06      |                       1    |
| meta-llama/Llama-3.2-3B-Instruct      | False                       |        1 |       0.04      |                       0.42 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | False                       |        5 |       0.06      |                       0.94 |
| Qwen/Qwen2.5-14B-Instruct             | False                       |        0 |       0         |                       0    |
| Qwen/Qwen2.5-3B-Instruct              | False                       |        5 |       0.015     |                       0.66 |
| Qwen/Qwen2.5-1.5B-Instruct            | False                       |        5 |       0         |                       0.48 |
| meta-llama/Llama-3.2-1B-Instruct      | True                        |        3 |       0.015     |                       1    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | True                        |        1 |       0.095     |                       1    |
| Qwen/Qwen2.5-1.5B-Instruct            | False                       |        0 |       0         |                       0    |
| Qwen/Qwen2.5-3B-Instruct              | True                        |        3 |       0.04      |                       1    |
| meta-llama/Llama-3.2-1B-Instruct      | True                        |        5 |       0.02      |                       1    |
| Qwen/Qwen2.5-3B-Instruct              | False                       |        3 |       0.02      |                       0.86 |
| meta-llama/Llama-3.2-3B-Instruct      | True                        |        1 |       0.0555556 |                       1    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | True                        |        3 |       0.065     |                       1    |
| meta-llama/Llama-3.2-1B-Instruct      | False                       |        5 |       0.025     |                       0.16 |
| Qwen/Qwen2.5-1.5B-Instruct            | True                        |        3 |       0.02      |                       1    |
| Qwen/Qwen2.5-14B-Instruct             | False                       |        5 |       0.085     |                       0.84 |
| meta-llama/Meta-Llama-3.1-8B-Instruct | False                       |        1 |       0.16      |                       0.68 |
| Qwen/Qwen2.5-14B-Instruct             | True                        |        0 |       0.14      |                       1    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | False                       |        0 |       0         |                       0    |
| Qwen/Qwen2.5-1.5B-Instruct            | False                       |        1 |       0         |                       0.2  |
| meta-llama/Llama-3.2-1B-Instruct      | True                        |        1 |       0.02      |                       1    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | True                        |        0 |       0.12      |                       0.96 |
| Qwen/Qwen2.5-3B-Instruct              | True                        |        1 |       0.05      |                       1    |
| meta-llama/Llama-3.2-3B-Instruct      | False                       |        0 |       0         |                       0    |
| meta-llama/Llama-3.2-3B-Instruct      | True                        |        3 |       0.01      |                       1    |
| Qwen/Qwen2.5-14B-Instruct             | False                       |        3 |       0.07      |                       0.64 |
| Qwen/Qwen2.5-3B-Instruct              | True                        |        5 |       0.025     |                       1    |
| meta-llama/Llama-3.2-3B-Instruct      | True                        |        5 |       0.01      |                       1    |
| meta-llama/Llama-3.2-3B-Instruct      | False                       |        3 |       0.055     |                       0.44 |
| meta-llama/Llama-3.2-1B-Instruct      | True                        |        0 |       0.015     |                       0.94 |
| meta-llama/Llama-3.2-1B-Instruct      | False                       |        0 |       0         |                       0    |
| Qwen/Qwen2.5-1.5B-Instruct            | True                        |        1 |       0         |                       1    |
| meta-llama/Llama-3.2-1B-Instruct      | False                       |        1 |       0.005     |                       0.06 |
| meta-llama/Llama-3.2-1B-Instruct      | False                       |        3 |       0.01      |                       0.14 |
| Qwen/Qwen2.5-14B-Instruct             | True                        |        5 |       0.055     |                       1    |
| meta-llama/Llama-3.2-3B-Instruct      | True                        |        0 |       0.065     |                       0.94 |
| Qwen/Qwen2.5-14B-Instruct             | True                        |        1 |       0.115     |                       0.98 |
| Qwen/Qwen2.5-1.5B-Instruct            | True                        |        5 |       0.005     |                       1    |
| Qwen/Qwen2.5-14B-Instruct             | False                       |        1 |       0         |                       0    |
| meta-llama/Meta-Llama-3.1-8B-Instruct | False                       |        3 |       0.08      |                       0.96 |

### Analysis

There were some trends in particular I was curious about, so here are a bunch of covariate analyses using spearman's rank

| Variable 1 | Variable 2 | Correlation | P Value |
|:-----------|:-----------|:-----------|:-----------|
| Model Params | Accuracy | 0.531 | 1e-4 |
| Model Params | Validity | 0.084 | 0.561|
| K-shot | Accuracy | 0.131 | 0.373 |
| K-shot | Validity | 0.301 | 0.037 |
| Structured | Accuracy | 0.279 | 0.055 |
| Structured | Validity | 0.885 | 1e-17 |

Due to the limited data and use of covariate techniques I wouldn't draw any conclusions about trends. There are also confounding variables
that aren't really discussed within the context of this analysis (# param probably isn't as meaningful between model families, e.g., 8B Llama3 outperformed
Gwen). Even for variables like K-shot, performance for a given K might also vary with the model.

<!-- TODO: Add an analysis for multivariate other?? -->

## Conclusion and Next Steps

Current results are rather poor, with the best model getting only 16% (32/200) of the groups correct. While there are tweaks that could be done to the experiments above such as:
1. Use of ChatGPT
2. Higher values of K
3. Use of a regex for guided generation (requires a different library or custom code
4. Prompt optimization, likely not manually but rather through use of prompt optimization tools like [DSPy](https://github.com/stanfordnlp/dspy) or [TextGrad](https://github.com/zou-group/textgrad)

I would rather focus on fundamentally different approaches that are more efficient and potentially leverage data structures to solve the problem.


Next attempt will likely be a combination of framing the problem as classification (whether two ore more words are related) and use
of a dynamic programming algorithm to pick which groups of four is most likely, potentially with additional checks for validity. This would enable
batching and potentially even use of smaller models.