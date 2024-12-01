# Fun with Words: A Foray into Solving NYT Connections via Decomposition (1.5/N)

** Currently A WIP 12/1**

As discussed in the [last post on "solving" NYT connections](https://felixlabelle.com/2024/10/30/fun-with-words-nyt-connections.html), 
this post will focus on the use of explicit decomposition rather than further improving generative solutions to NYT connections. No need to have the model find the structure on its own, instead make the problem a "simpler" one and use search to find
the best fitting solution.
The problem is divided into pairwise (word) [similarity](https://felixlabelle.com/2023/11/18/discussion-about-text-similarity.html#defining-similarity)
and then those similarities are in turn used to find the best groups of four using a search algorithm.


My inclination before running these experiments was that this approach would benefit both performance and speed. In spite of all the improvements
in models the recent years I don't think they have the ability to "reason". Instead I think by guiding them through with use of explicit structure we can leverage
the knowledge they capture about the world without being limited by their inability to "reason".
Another potential benefit is increasing the ease of collecting training data by transforming the task from a one where puzzles need to be 
generated to one where word pairs need to have potential links identified.

## Relevant Work

[A paper introducing NYT connections dataset was published in EMNLP](https://arxiv.org/html/2406.11012v3) 1-2 weeks ago as of writing. I hadn't seen it pre-EMNLP (although it's been on Arxiv since July),
so this post will reuse last post's dataset. In an upcoming post on NYT connections I plan on using their subset of the connections problems, their taxonomy, and their metrics to allow
1. comparison to closed source models like GPT-4
2. comparison to human performance
3. finer-grained analysis of model outputs

## Experiment

The experiment consists of two parts
1. classifying word pairs using prompted models
2. using search to find the most probably 4 groups of 4

The following subsections explore the setup of the experiments and different design decisions.
The code used to run the experiments can be [found here](https://github.com/FelixLabelle/connections_solver). 

### Dataset

[This NYT connections repo by "Eyefyre"](https://github.com/Eyefyre/NYT-Connections-Answers) was used with an October 12th cutoff was used to make the results comparable
to the generative approach.

### Design Decisions


#### Not Turn-based

Unlike the game this version is not turn based, rather the model has a single shot to find all 4 groups. While a turn-based approach could be interesting (it enables for use of 
multi-turn strategies and self-correction), it's been ignored for now to simplify things.

#### Sample Size

A sample size of 50 was used to maintain compatibility with the last experiments. The same items were used.

#### Seed

Fixed seeds were set for pretty much all random operations such as shuffle and passed in a seed wherever possible. Currently all operations
use the same seed, though that may change to enable for multiple generation seeds.

The seed was 42, because 42.

#### Prompted Models for Classification 

To classify if two words are similar prompting of LLMs was used exclusively because
1. types of relationships required by connections might limit viable models
2. reuse of the same models makes comparison to the previous post easier.
3. limited training data

To go into more depth about point 1., the types of relationships captured in NYT connections go beyond word association. [Samadarshi et al.](https://arxiv.org/html/2406.11012v3) propose a taxonomy on the types of groupings in NYT connections:

![Grouping taxonomy proposed by Samadarshi et al.](/images/nyt_taxonomy_similarity.png)

At a quick glance, some of these categories can't be handled by word embeddings. E.g., polysemy; word embeddings aren't contextual so they can't give multiple representations to a given token. More likely than 
not word embeddings will fail at associations requiring this relationship (my guess is unless it's their most common usage).

Similarly I'm not sure how well word embeddings capture knowledge. Not sure orange objects will be grouped similarly in an embedding space, for example
carrot and goldfish give a similarity of 0.23 which would mean this pair is considered unlikely although both have a relationship (they are all organge).
<!-- TODO: Run experiment here and add details -->
<!-- http://epsilon-it.utu.fi/wv_demo/ -->

LLMs seem like a better fit since they do appear to capture some ["knowledge"]().
<!-- TODO: Find a good citation-->

#### Extracting Labels

To reliably extract binary classes  from the prompted LLMs masking was used.
Any token that can be turned into `yes` or `no` by lower casing or stripping leading & trailing spaces is considered a valid output.
The indices of all the yes and no tokens are used as a mask.
The logits are used to calculate a probability distribution. The yes/no mask is applied and the resulting probabilities
are normalized (they sum to 1).
All the probabilities for the yes tokens are summed together and that is the probability of two words being similar. This is 
what is used for search.

{% raw %}
```
yes_idxs = [idx for word,idx in tokenizer.vocab.items() if word.replace('Ġ','').lower().strip() == "yes"]
no_idxs = [idx for word,idx in tokenizer.vocab.items() if word.replace('Ġ','').lower().strip() == "no"]
yes_no_idxs = yes_idxs + no_idxs

## SKIPPED LINES

# This might seem odd, but the Qwen models have a mismatch
# between vocab and output size
if 'Qwen' in model_id:
	# NOTE: Hacky way of getting real embedding size vs vocab size
	# Qwen pads their embeddings to get a power of 2 divisble number and it's incompatible with the way HF stores vocab..
	# I tried different ways of extracting the output size, but couldn't get it to work consistently across resolutions, not sure why
	mask_size = len(tokenizer)
	p_delta = min([dim-mask_size for p in model.parameters() for dim in p.shape],key=abs)
	mask_size += p_delta
else:
	mask_size = len(tokenizer)
            
## SKIPPED LINES
	
mask = torch.zeros(mask_size)
mask[yes_no_idxs] = 1

output = model.generate(
	**inputs,
	max_new_tokens=1,  # Set based on expected response length
	do_sample=False,
	return_dict_in_generate=True,
	output_scores=True,
	top_p=None,
	temperature=None,
)
model_outputs.append(output['scores'][0])

output_probs = torch.softmax(torch.vstack(model_outputs),dim=1).to("cpu")
premask_probs = output_probs[:,yes_no_idxs].sum(axis=1)
masked_probs = output_probs * mask
masked_probs /= masked_probs.sum(axis=1).reshape(-1,1).tile(1,masked_probs.shape[1])
yes_prob = masked_probs[:,yes_idxs].sum(axis=1)
```
{% endraw %}

#### Similarity Aggregation Method

The model's predictions and resulting confidence are not order invariant. The range of difference in order is ... for ...
The binary classifier analysis provides more details on the spread.

Aggregation methods are used to overcome this limitation. Different aggregation methods were implemented;
`aggregation_dict = {"min" : min,"max" : max,"mean": mean, "first" : lambda x: x[0], "last" : lambda x:[-1]}`

Only `max` was used to minimize the number of hyperparameters tried.

#### Search

Search was not a big focus of this work, a local search algorithm was implemented and used.
The global score of a given example is given by 
```
def evaluate(words,sims,generation_chunk_size):
    return np.prod([sims[idx_combination] for group in batcher(words,4) for idx_combination in generate_keys(group,generation_chunk_size)])
```

where `words` is a list of 16 words. Each group of four within it represents a model prediction. Sims is a dictionary which uses a tuple of words
as indices and model confidence as the output. 
An initial list of `'k'
random groupings are first generated then scored. Pairwise permutations are conducted for each input.
`search_type` can be `random` or `greedy`. For `greedy` the top `k` inputs are scored and then each permuted.
`random` picks `k` random inputs. Annealing strategies were not implemented nor tested. After computing the scores
of these permutations the top `k` are kept. The algorithm continues until `patience` turns have passed without improvement.


{% raw %}
```
def local_search(sims, generation_chunk_size, k =10, patience=0, search_type="greedy",top_p = 1.0, search_random_top_k=-1):
    idx_tuples_to_search = [tuple(random.sample(words, 16)) for _ in range(k)]
    idx_scores = [evaluate(idx_tuple_to_search,sims,generation_chunk_size) for idx_tuple_to_search in idx_tuples_to_search]
    max_score = max(idx_scores)
    idx_score_mapping = {idx_tuple_to_search: idx_score for idx_tuple_to_search, idx_score in zip(idx_tuples_to_search,idx_scores)}
    searching = True
    turns_without_improvement = 0
    def swap_pos(tpl,idx1,idx2):
        lst = list(tpl)
        tmp = lst[idx1]
        lst[idx1] = lst[idx2]
        lst[idx2] = tmp
        return tuple(lst)
        
    while searching:
        # TODO: adapt this for 4 position
        idx_permutation_tuples = [swap_pos(idx_tuple_to_search,idx1,idx2) for (idx1,idx2) in generate_keys(range(16),generation_chunk_size) for idx_tuple_to_search in idx_tuples_to_search] # TODO: Create iteration tuples from idxs_to_search, remove dupes
        for idx_permutation_tuple in idx_permutation_tuples:
            if idx_permutation_tuple in idx_score_mapping:
                pass
            else:
                idx_score_mapping[idx_permutation_tuple] = evaluate(idx_permutation_tuple,sims,generation_chunk_size)
        
        idx_score_mapping = dict(sorted(idx_score_mapping.items(), key=lambda item: item[1],reverse=True))
        # TODO: Look at other selection methods, currently greedy by default
        if search_type == "greedy":
            new_idx_tuples_to_search, idx_scores = zip(*[item for item, _ in zip(idx_score_mapping.items(),range(k))])
        elif search_type == "random":
            #import pdb;pdb.set_trace()
            filtered_items = idx_score_mapping.items()
            if top_p < 1.0:
                raise NotImplementedError()
            if search_random_top_k > 0:
                filtered_items = [tpl for tpl,_ in zip(idx_score_mapping.items(),range(search_random_top_k))]
            new_idx_tuples_to_search, idx_scores = zip(*random.sample(filtered_items,k))
        else:
            raise NotImplementedError(f"Search type {search_type} not implemented")
            
        max_new_score = max(idx_scores)
        if set(idx_tuples_to_search) == set(new_idx_tuples_to_search): # CHECK IF LISTS ARE OVERLAPPED, MAYBE USE SET
            searching=False
        elif max_new_score > max_score:
            turns_without_improvement = 0
            max_score = max_new_score
            idx_tuples_to_search = new_idx_tuples_to_search
        else:
            turns_without_improvement += 1
            if turns_without_improvement > patience:
                searching = False
        
    # TODO: Extract max score from idx_score_mapping (we could search other)
    return idx_tuples_to_search[0],idx_scores[0]
```
{% endraw %}

For all the experiments conducted the settings were:
```
k=10
search_type="greedy"
search_patience=0
```

These settings were determined after a small pilot using `Llama3.1-8B`. While better results were achieved by increasing `k` and `patience` up to 5, there was a point of diminishing returns. Given the long run times 
already I decided to leave this be for now.
Currently results are precomputed so search can be redone after the fact. I may explore different optimization methods later using this data.

#### Use of Pairs
<!-- TODO: Add word invariance reference -->

Originally both groups of 2 and 4 where going to be used to calculate similarity for a group. However combinatorics, the lack of word invariance in LLMs, and limited engineering efforts
made groups of 4 not computationally viable. Groups of 4 require 16 choose 4 (1820) GPU inferences at a minimum. However, this assumes that 
the similarity given to a particular group is order invariant. In other words:
sim((1,2,3,4)) = sim((4,3,2,1))

However this isn't the case (see binary results analysis below). That means to get a good estimate on the "true similarity" we need to try multiple permutations of a given group
and aggregate them. If we try all the 4 sized permutations of 16 we get that there are 43680 permutations. Each of those is a GPU call and the current approach calculates all the similarities upfront and then does search using those. 

I originally tried limiting the number of permutations tried to 2-3, but this is still brutally slow. Realistically search would need to select which groups to calculate.

### Hyper-parameters (Independent Variables)

Below is the list of hyper-parameters that were varied and their values. The experiment runs essentially a grid search of the following hyper-parameters. See [generate_experiments.py]().


#### Models

To preserve compatibility with the previous experiments only Qwen and Llama3 models were used. Qwen 7B as a comparable 
model to Llama3 8B in analyses related to the role of param-size and model family.

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

#### Quantization

Unlike the generative experiments where only 4 bit models were used, the decomposition approach leveraged 4,8 and 16-bit models as well. The reasoning was two-fold
1. Less likely to run into OOMs or take an excessive amount of time due to limited output length
2. Quantization may have an impact on the model's "confidence" and skew results (this is related to calibration I think)
<!-- is 2 true? If so is this a good citation https://aclanthology.org/2024.acl-long.544/? -->

#### Prompts

Two prompts were used, themed and default. They are:

{% raw %}
```
default_system_prompt = "Given two words, give a yes or no answer as to whether or not the words have a relationship."
theme_aware_system_prompt = """Given two words, give a yes or no answer as to whether or not the words have a relationship.
The types of relationships can include, but are not limited to
1. Synonyms
2. Homophones
3. Sharing a leading or train word
4. Some common usage
5. Names of things in a similar group
6. Physical similarities
7. Anagrams

You are an expert linguistic, so please be as confident as possible. If there are no obvious connections, say no"""

system_prompt_dict = {
"default" : default_system_prompt,
"themed": theme_aware_system_prompt,
}
```
{% endraw %}


#### K-Shot

K-shot examples (other word pairs) were also appended to the prompt to improve performance. Different k-shot values were used.
`k_shot_options = [0, 1, 3, 5]`

K-shot examples are picked at random from the connections dataset (or sample) excluding the current item and added afterwards.

{% raw %}
```
if k_shot > 0:
	# TODO: Implement k shot
	examples = nyt_connections_data[:datum_idx] + nyt_connections_data[datum_idx+1:]
	k_shot_examples = random.Random(seed).sample(examples,k_shot)
	for k_shot_example in k_shot_examples:
		k_shot_connections_words = [word for item in k_shot_example['answers'] for word in item['members']]
		formatted_example = {"groups" : [{"words" : group['members'],"theme" : group['group']} for group in k_shot_example["answers"]]}
		messages.append({"role" : "user", "content" : f"Your words are {";".join(k_shot_connections_words)}. Good luck!"})
		messages.append({"role" : "assistant", "content" : json.dumps(formatted_example)})
```
{% endraw %}

### Metrics (Dependent Variables)

Only one metric was implemented, mean accuracy i.e., percentage of groups the model got correct. This approach isn't difficulty aware, nor is it taxonomy aware.
These metrics were ignored for now to be consistent with the previous post, but will be added in an upcoming post.

## Results

Below are the results. Any missing results are due to the experiment erroring out. A small number failed due to OOMs. Best effort was made to 
run every test point, but some slipped by.

| model_id                              |   param_count |   prompt_k | prompt_version   |   resolution |   mean_accuracy |
|:--------------------------------------|--------------:|-----------:|:-----------------|-------------:|----------------:|
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | default          |            4 |           0.015 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | default          |            8 |           0     |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | default          |           16 |           0     |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | default          |            4 |           0.05  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | default          |            8 |           0.03  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | default          |           16 |           0.06  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | default          |            4 |           0.085 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | default          |            8 |           0.065 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | default          |           16 |           0.13  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | default          |            4 |           0.115 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | default          |            8 |           0.095 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | default          |           16 |           0.145 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | themed           |            4 |           0     |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | themed           |            8 |           0.01  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          0 | themed           |           16 |           0.005 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | themed           |            4 |           0.04  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | themed           |            8 |           0.045 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          1 | themed           |           16 |           0.08  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | themed           |            4 |           0.095 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | themed           |            8 |           0.085 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          3 | themed           |           16 |           0.145 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | themed           |            4 |           0.105 |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | themed           |            8 |           0.12  |
| meta-llama/Llama-3.2-1B-Instruct      |           1   |          5 | themed           |           16 |           0.175 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | default          |            4 |           0.12  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | default          |            8 |           0.14  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | default          |           16 |           0.15  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | default          |            4 |           0.16  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | default          |            8 |           0.165 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | default          |           16 |           0.165 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | default          |            4 |           0.16  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | default          |            8 |           0.19  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | default          |           16 |           0.185 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | default          |            4 |           0.165 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | default          |            8 |           0.16  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | default          |           16 |           0.17  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | themed           |            4 |           0.105 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | themed           |            8 |           0.095 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          0 | themed           |           16 |           0.11  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | themed           |            4 |           0.16  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | themed           |            8 |           0.21  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          1 | themed           |           16 |           0.22  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | themed           |            4 |           0.175 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | themed           |            8 |           0.215 |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          3 | themed           |           16 |           0.22  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | themed           |            4 |           0.15  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | themed           |            8 |           0.22  |
| Qwen/Qwen2.5-1.5B-Instruct            |           1.5 |          5 | themed           |           16 |           0.215 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | default          |            4 |           0.16  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | default          |            8 |           0.13  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | default          |           16 |           0.14  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | default          |            4 |           0.21  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | default          |            8 |           0.2   |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | default          |           16 |           0.19  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | default          |            4 |           0.205 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | default          |            8 |           0.2   |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | default          |           16 |           0.21  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          5 | default          |            4 |           0.22  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          5 | default          |            8 |           0.18  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          5 | default          |           16 |           0.185 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | themed           |            4 |           0.175 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | themed           |            8 |           0.16  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          0 | themed           |           16 |           0.205 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | themed           |            4 |           0.24  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | themed           |            8 |           0.2   |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          1 | themed           |           16 |           0.23  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | themed           |            4 |           0.24  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | themed           |            8 |           0.22  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          3 | themed           |           16 |           0.22  |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          5 | themed           |            4 |           0.225 |
| Qwen/Qwen2.5-3B-Instruct              |           3   |          5 | themed           |           16 |           0.22  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | default          |            4 |           0.115 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | default          |            8 |           0.055 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | default          |           16 |           0.155 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | default          |            4 |           0.165 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | default          |            8 |           0.13  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | default          |           16 |           0.125 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | default          |            4 |           0.145 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | default          |            8 |           0.15  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | default          |           16 |           0.16  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | default          |            4 |           0.165 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | default          |            8 |           0.145 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | default          |           16 |           0.18  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | themed           |            4 |           0.155 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | themed           |            8 |           0.05  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          0 | themed           |           16 |           0.13  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | themed           |            4 |           0.19  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | themed           |            8 |           0.175 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          1 | themed           |           16 |           0.185 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | themed           |            4 |           0.18  |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | themed           |            8 |           0.185 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          3 | themed           |           16 |           0.2   |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | themed           |            4 |           0.205 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | themed           |            8 |           0.195 |
| meta-llama/Llama-3.2-3B-Instruct      |           3   |          5 | themed           |           16 |           0.23  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | default          |            4 |           0.175 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | default          |            8 |           0.185 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | default          |           16 |           0.16  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | default          |            4 |           0.2   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | default          |            8 |           0.205 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | default          |           16 |           0.21  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | default          |            4 |           0.21  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | default          |            8 |           0.185 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | default          |           16 |           0.155 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | default          |            4 |           0.2   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | default          |            8 |           0.19  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | default          |           16 |           0.17  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | themed           |            4 |           0.22  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | themed           |            8 |           0.205 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          0 | themed           |           16 |           0.2   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | themed           |            4 |           0.19  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | themed           |            8 |           0.235 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          1 | themed           |           16 |           0.24  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | themed           |            4 |           0.23  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | themed           |            8 |           0.215 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          3 | themed           |           16 |           0.29  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | themed           |            4 |           0.2   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | themed           |            8 |           0.235 |
| meta-llama/Meta-Llama-3.1-8B-Instruct |           8   |          5 | themed           |           16 |           0.25  |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          0 | default          |            4 |           0.175 |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          1 | default          |            4 |           0.245 |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          3 | default          |            4 |           0.19  |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          5 | default          |            4 |           0.225 |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          0 | themed           |            4 |           0.285 |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          0 | themed           |           16 |           0.24  |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          1 | themed           |            4 |           0.3   |
| Qwen/Qwen2.5-14B-Instruct             |          14   |          3 | themed           |            4 |           0.265 |

## Analysis

Two analyses were conducted, one for performance trends and the other comparing structured approaches to generative approaches. Originally a third analysis was planned
to see the correlation between word similarity performance and global performance. This may be added later or part of another post down the road.

### Performance Trends

This analysis has about 157 experiments, making analysis more meaningful than last time where just 48 experiments were present.
The independent variables studied are parameter count, the model family (llama (0) vs qwen (1)), the prompt used,
resolution, and number of examples. Two analyses used to measure correlation:
1. spearman's rank
2. coefficient analysis using linear regression

#### Spearman's Rank Table

| Independent Variable | Dependent Variable | Statistic | p-value                |
|-----------------------|--------------------|-----------|------------------------|
| `param_count`        | `mean_accuracy`   | 0.6646346 | \(6.206 \times 10^{-19}\) |
| `prompt_k`           | `mean_accuracy`   | 0.2580035 | 0.00225               |
| `resolution`         | `mean_accuracy`   | -0.046676 | 0.58670               |
| `prompt_version`     | `mean_accuracy`   | 0.3167554 | \(1.536 \times 10^{-4}\) |
| `model_family`       | `mean_accuracy`   | 0.3691301 | \(8.38 \times 10^{-6}\) |


#### Linear Regression Tble

| Independent Variable | Dependent Variable | Statistic |
|-----------------------|--------------------|-----------|
| `param_count`        | `mean_accuracy`   | 0.5843    |
| `prompt_k`           | `mean_accuracy`   | 0.3155    |
| `resolution`         | `mean_accuracy`   | 0.1312    |
| `prompt_version`     | `mean_accuracy`   | 0.2281    |
| `model_family`       | `mean_accuracy`   | 0.3849    |  


#### Analysis

It looks like parameter count is again the biggest factor when it comes to improving performance, although unlike last time 
the number of k examples seems help rather than hurt. Not sure why that is.

Qwen model's seem to do better, but there are some confounding factors. The fact the largest model is a Qwen model so this may be throwing the analysis off.

Resolution's role is odd too. In the multivariate analysis it seems to matter (the least, but still) unlike in the correlation analysis where it seems to play
any role.

<!--
#### Correlation between Binary Performance and Downstream Performance

Not shockingly the better the precision and recall on binary performance did the better the downstream model did.

The correlation is ...
-->

### Generative vs Structured Prediction

In terms of performance, structured prediction is a clear winner. To compare the two experiments results were lined up based on
1. Shared resolution
2. Model used
3. K-shot
4. Prompt used (Default-only)
This created 48 paired scores. When comparing the generative experiments one-to-one in this way and structured prediction experiments
the mean average accuracy is significantly higher (12.5% absolute improvement on average) for structured prediction. The p-value of this result is 0.0002 and 
was calculated using resampling with the statistic being the mean of the difference of values between experiments.

Both small and large models seem to benefit from task decomposition. Given the limited data (48 samples), this might be hard to prove conclusively,
but at a glance looks to be true. Moreover the models no longer generate invalid outputs which may one boost performance, but two improve reliability.


In terms of time the results are less conclusive. For smaller models and larger models under smaller k-shot values experiment times are any from 95%-50% less. But for the larger models experiments are rather slow
with one taking 20 times longer.
I think this is due to the larger models and longer queries being split between CPU and GPU which negated the advantage of larger batch sizes. The raw results for the comparisons are
presented below. `time_ratio` is the percentage of time the structured approach took compared to the generative approach (structured/gen). `accuracy_delta` is given by `mean_accuracy_structured` - `mean_accuracy_generative`.

| model_id                              |   k_shot |   total_run_time_seconds_generative |   total_run_time_seconds_structured |   mean_accuracy_generative |   mean_accuracy_structured |   accuracy_delta |   time_ratio |
|:--------------------------------------|---------:|------------------------------------:|------------------------------------:|---------------------------:|---------------------------:|-----------------:|-------------:|
| Qwen/Qwen2.5-1.5B-Instruct            |        0 |                            1482.99  |                             60.7417 |                  0.02      |                      0.12  |         0.1      |    0.0409589 |
| Qwen/Qwen2.5-1.5B-Instruct            |        0 |                           19610.5   |                             60.7417 |                  0         |                      0.12  |         0.12     |    0.0030974 |
| Qwen/Qwen2.5-1.5B-Instruct            |        1 |                             679.193 |                            137.074  |                  0         |                      0.16  |         0.16     |    0.201818  |
| Qwen/Qwen2.5-1.5B-Instruct            |        1 |                            1403.12  |                            137.074  |                  0         |                      0.16  |         0.16     |    0.0976917 |
| Qwen/Qwen2.5-1.5B-Instruct            |        3 |                            1429.25  |                            227.357  |                  0.02      |                      0.16  |         0.14     |    0.159074  |
| Qwen/Qwen2.5-1.5B-Instruct            |        3 |                             611.559 |                            227.357  |                  0         |                      0.16  |         0.16     |    0.371766  |
| Qwen/Qwen2.5-1.5B-Instruct            |        5 |                            1430.22  |                            321.217  |                  0.005     |                      0.165 |         0.16     |    0.224593  |
| Qwen/Qwen2.5-1.5B-Instruct            |        5 |                             612.976 |                            321.217  |                  0         |                      0.165 |         0.165    |    0.524028  |
| Qwen/Qwen2.5-14B-Instruct             |        0 |                            2261.71  |                            424.713  |                  0.14      |                      0.175 |         0.035    |    0.187784  |
| Qwen/Qwen2.5-14B-Instruct             |        0 |                            4206.25  |                            424.713  |                  0         |                      0.175 |         0.175    |    0.100972  |
| Qwen/Qwen2.5-14B-Instruct             |        1 |                            2016.38  |                            796.086  |                  0.115     |                      0.245 |         0.13     |    0.39481   |
| Qwen/Qwen2.5-14B-Instruct             |        1 |                            3352.15  |                            796.086  |                  0         |                      0.245 |         0.245    |    0.237485  |
| Qwen/Qwen2.5-14B-Instruct             |        3 |                            1780.46  |                          35432.8    |                  0.07      |                      0.19  |         0.12     |   19.9009    |
| Qwen/Qwen2.5-14B-Instruct             |        3 |                            1968.27  |                          35432.8    |                  0.06      |                      0.19  |         0.13     |   18.002     |
| Qwen/Qwen2.5-14B-Instruct             |        5 |                            1524.58  |                           2431.42   |                  0.085     |                      0.225 |         0.14     |    1.59481   |
| Qwen/Qwen2.5-14B-Instruct             |        5 |                            1945.69  |                           2431.42   |                  0.055     |                      0.225 |         0.17     |    1.24964   |
| Qwen/Qwen2.5-3B-Instruct              |        0 |                            1596.94  |                            101.313  |                  0.035     |                      0.16  |         0.125    |    0.0634416 |
| Qwen/Qwen2.5-3B-Instruct              |        0 |                            4035.11  |                            101.313  |                  0         |                      0.16  |         0.16     |    0.0251078 |
| Qwen/Qwen2.5-3B-Instruct              |        1 |                            1746.05  |                            230.215  |                  0.05      |                      0.21  |         0.16     |    0.131849  |
| Qwen/Qwen2.5-3B-Instruct              |        1 |                             898.836 |                            230.215  |                  0.04      |                      0.21  |         0.17     |    0.256126  |
| Qwen/Qwen2.5-3B-Instruct              |        3 |                            1840.21  |                            376.323  |                  0.04      |                      0.205 |         0.165    |    0.204499  |
| Qwen/Qwen2.5-3B-Instruct              |        3 |                             871.644 |                            376.323  |                  0.02      |                      0.205 |         0.185    |    0.431739  |
| Qwen/Qwen2.5-3B-Instruct              |        5 |                            2060.8   |                            578.724  |                  0.025     |                      0.22  |         0.195    |    0.280826  |
| Qwen/Qwen2.5-3B-Instruct              |        5 |                             885.635 |                            578.724  |                  0.015     |                      0.22  |         0.205    |    0.653457  |
| meta-llama/Llama-3.2-1B-Instruct      |        0 |                            1060.83  |                             65.9427 |                  0.015     |                      0.015 |         0        |    0.0621613 |
| meta-llama/Llama-3.2-1B-Instruct      |        0 |                             808.135 |                             65.9427 |                  0         |                      0.015 |         0.015    |    0.0815986 |
| meta-llama/Llama-3.2-1B-Instruct      |        1 |                            1048.55  |                             95.059  |                  0.02      |                      0.045 |         0.025    |    0.0906575 |
| meta-llama/Llama-3.2-1B-Instruct      |        1 |                             701.163 |                             95.059  |                  0.005     |                      0.045 |         0.04     |    0.135573  |
| meta-llama/Llama-3.2-1B-Instruct      |        3 |                            1053.63  |                            160.37   |                  0.015     |                      0.105 |         0.09     |    0.152207  |
| meta-llama/Llama-3.2-1B-Instruct      |        3 |                             484.118 |                            160.37   |                  0.01      |                      0.105 |         0.095    |    0.331263  |
| meta-llama/Llama-3.2-1B-Instruct      |        5 |                             502.513 |                            234.423  |                  0.025     |                      0.125 |         0.1      |    0.466501  |
| meta-llama/Llama-3.2-1B-Instruct      |        5 |                            1042.35  |                            234.423  |                  0.02      |                      0.125 |         0.105    |    0.224899  |
| meta-llama/Llama-3.2-3B-Instruct      |        0 |                            1302.76  |                            105.559  |                  0.065     |                      0.115 |         0.05     |    0.081027  |
| meta-llama/Llama-3.2-3B-Instruct      |        0 |                            1245.5   |                            105.559  |                  0         |                      0.115 |         0.115    |    0.0847519 |
| meta-llama/Llama-3.2-3B-Instruct      |        1 |                            9722.71  |                            187.601  |                  0.0555556 |                      0.165 |         0.109444 |    0.0192952 |
| meta-llama/Llama-3.2-3B-Instruct      |        1 |                             858.433 |                            187.601  |                  0.04      |                      0.165 |         0.125    |    0.218539  |
| meta-llama/Llama-3.2-3B-Instruct      |        3 |                             847.365 |                            398.698  |                  0.055     |                      0.145 |         0.09     |    0.470515  |
| meta-llama/Llama-3.2-3B-Instruct      |        3 |                            1324.7   |                            398.698  |                  0.01      |                      0.145 |         0.135    |    0.300972  |
| meta-llama/Llama-3.2-3B-Instruct      |        5 |                             789.329 |                            680.728  |                  0.06      |                      0.165 |         0.105    |    0.862413  |
| meta-llama/Llama-3.2-3B-Instruct      |        5 |                            1292.73  |                            680.728  |                  0.01      |                      0.165 |         0.155    |    0.526583  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        0 |                            1422.47  |                            219.826  |                  0.12      |                      0.175 |         0.055    |    0.154538  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        0 |                            2102.29  |                            219.826  |                  0         |                      0.175 |         0.175    |    0.104565  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        1 |                             861.46  |                            378.684  |                  0.16      |                      0.2   |         0.04     |    0.439583  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        1 |                            1359.49  |                            378.684  |                  0.095     |                      0.2   |         0.105    |    0.278548  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        3 |                             671.307 |                            884.927  |                  0.08      |                      0.21  |         0.13     |    1.31822   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        3 |                            1363.04  |                            884.927  |                  0.065     |                      0.21  |         0.145    |    0.649232  |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        5 |                             675.371 |                           1360.02   |                  0.06      |                      0.2   |         0.14     |    2.01374   |
| meta-llama/Meta-Llama-3.1-8B-Instruct |        5 |                            1433.59  |                           1360.02   |                  0.02      |                      0.2   |         0.18     |    0.948679  |

## Conclusion and Next Steps

Long story short task decomposition and structured prediction does still help this generation of models when solving NYT connections. There are
both performance and speed gains, although speed gains come from batching which may be hampered if your model doesn't fit on GPU. Another advantage of this approach is that you can more easily gather data for the sub-problem than the larger problem. Rather than synthetically create
groupings of 16 with 4 themes, training or evaluation data can consistent of word pairs.

Considering this is only one task mostly with "smaller" models, the conclusion that we can draw about model performance are somewhat limited. It does appear that structure 

While performance has improved drastically (nearly doubling even for the larger models) it is still lacking IMO. The scores are similar to those for novices in
Samadarshi et al. (although we don't use the same dataset so this may not hold). In a future post the following things will be investigated:
1. Investigate different search techniques
2. Integrate generation and search for efficiency
3. Explore additional hyper-parameters
	1. Larger models (Qwen models above 14B parameters)
	2. Taxonomy based prompts
	3. Larger k-values
4. Using Samadarshi et al.'s dataset and additional metrics

A longer term plan is to build a light model able to cheaply do classification tasks such as these with no additional training. Ideally even
smaller than 1B. I don't think
it's possible to create universally-useful representations. I think while pretrained approaches may generate representations that help across most if not all tasks,
there is a limit and at some point it is necessary to specialize. For this reason I'm focusing on ways of either searching for suitable representations or
using some way of "steering" model embeddings to this end.