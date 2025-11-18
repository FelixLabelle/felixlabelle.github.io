# Deep dive: On the Theoretical Limitations of Embedding-Based Retrieval

I recently came across [On the Theoretical Limitations of Embedding-Based Retrieval](https://arxiv.org/pdf/2508.21038), a paper which overlaps with my interests on
[better understanding design decisions when designing, training, or selecting models](https://felixlabelle.com/2025/01/01/direction_for_the_year.html).
I enjoy the type of work in this paper and wish to see more of it, but this blog post is going to take a more critical look at the results. Papers in this vein need to be rigorous
or at the very least well designed to be of use to practitioners.


Concurrently [this video critique of the same paper by Yannic Kilcher](https://www.youtube.com/watch?v=zKohTkN0Fyk) came out.
While there is overlap, the video has significantly more background on IR (which the reader is assumed to have for this post) and doesn't dive as deeply into the methodological issues
of the paper. Moreover, this blog will detail additional experiments and suggestions for how this 
work could be used in the future. I recommend watching the video first, especially for people unfamiliar with IR or who are newer to field.

Now on to the show. The blog will have the following structure. First discuss the paper's 3 contributions, followed by 
experiments I would have liked to see and analysis on future work.
1. proposing theoretical bounds on the minimal number of dimensions needed 
2. measuring empirical bound on dimensions needed for specific properties
3. practical experiments
4. experiments I'd like to see
5. implications and future directions

## Theoretical Bounds

Rather than go over the proof itself, the results will be presented and an interpretation of their implications provided.
A different notation from the paper will be used for simplicity and the terminology used, mathematical or otherwise 
will be defined below. Note these definitions that will be used through out the post:

$C$ is the collection of documents $c$. $c$ is used to avoid confusion with dimension $d$
$Q$ are the corpus of queries $q$
$|X|$ refers to size of an array $X$, concretely $|C|$ is the number of documents, $|Q|$ number of queries
$rel$ is a $|C|x|Q|$ matrix representing the relationships st $GT_(ij) = 1$ iff a query and document are related otherwise $GT_ij = 0$
$E_d$ is an embedding of the documents 
$E_q$ is an embedding of the queries
$\hat{rel}$ is an approximation of the ground truth relationships $R$ where $\hat{rel} = \cdot{E_d^T,E_q}$, in other words the result of cosine similarity between the output of an embedding model over the entire corpus
$rank$ refers to the rank of a matrix, ranking refers to act of ordering items, order will be used to refer to the order in which a models lists items
$k$ refers to k elements used. It will be used in the the context of top-k

With these definitions in hand, we move on to the proof itself. The paper lays out three properties of $rel$ that can be preserved and their relationship to $\hat{rel}$. These are:

1. row wise order given by $rank_rop A$ ; this represents the smallest number of dimensions that preserves the ground truth order between embeddings
2. local threshold rank given by $rank_rt A$; this is the smallest number of dimensions that preserves a certain distance $\tau_i$ for a given row $q$
3. global threshold rank given by $rank_gt A$; this is the smallest number of dimensions that preserves a certain distance $\tau$ for all rows

Property 1's utility is intuitive; what is the point of a ranking system that doesn't preserve order? As far as I can tell
properties 2 and 3 don't have an intuitive purpose, especially since we don't define the values of $\tau$ they should hold for.
From my understanding they are primarily used to establish a bound around the
number of dimensions $d$ required to capture these properties i.e., construct the proof.
The proof results in an inequality which ties all three of these properties together and finds bounds on them;
the bounds are defined using $sign_rank$, which takes $rel$, projects it to $\{-1,1\}$ by doing $rek - 1$ and takes the rank of that such that $sign_rank rel = rank(2 rel - 1)$.
In other words its still a binary except that zeros become -1 and 1 stays 1. 

$$sign\_rank(rel)-1 <= rank\_rop(rel) = rank_rt rel <= rank_gt rel < sign_rank(rel)$$


The long and short of it is that the sign rank matrix is the minimal size
required (look at both ends of the inequality) to preserve the 3 properties listed above.
The authors note that this means the minimum number of dimensions is task specific, more precisely that it is 
based solely on $rel$. I think there are two primary limitations for practical use that need to be further explored
1. Sign rank is difficult to compute. The authors briefly touch on this in the related works section, but essentially using Sign Rank (as of date of publishing) is a non-starter
2. Effect of changes to $rel$ w.r.t. to $d$ required, ideally in practice. I.E., does the minimum number of dimensions required change dramatically between p(y|x) shifts (e.g., new data, new split, new domain)

Spoiler, we will look at these two issues in the additional experiments. For now we will look 
at the method the authors propose to use to find the bound instead of sign rank.

## Empirical Bounds on Performance

Since sign rank can't be computed practically speaking, the authors propose an alternative method to establish an empirical bound they call "free embedding" optimization.
The setup is that for an embedding of size $d$ they try to find the maximum size of $|C|$ they can represent
regardless of the underlying task. I won't go into specifics of how they do this yet.

One question that naturally arises is how many queries should the dataset have and what should $rel$ look like?
The solution the authors arrive at is that every single top-k pair of documents needs to be represented. This leads to an interesting issue, which is that there are $|C| choose $k$ queries. This leads to very large values of $|Q|$,
as the authors point out. To minimize the combinatorial growth, they only evaluate small values of $d$ ([4,45]) and $k$ (2).
Using the 41 values calculated, a 3rd degree polynomial is fit.

$f(d) = −10.5322 + 4.0309 d + 0.0520 d^2 + 0.0037 d^3$

Using this polynomial, the max top-2 an embedding is extrapolated. Values found are:

| num dimensions | num items |
|------------|---------|
| 512        | 500k   |
| 768        | 1.7M   |
| 1,024      | 4M     |
| 3,072      | 107M   |
| 4,096      | 250M   |


### Methodological Issues

#### Moving Goal Posts

The paper focuses on "web-scale" data and this term is not well-defined in the paper or in general. Moreover, the scales tackled according to their estimates are 
rather large (250M documents with the correct top 2 with 4096 embeddings, upper bound of 350M). Even "just" 250M documents will cover most use cases assuming the value is correct (see below).
The statement that single vector embeddings are limited, based on these estimates, feels a bit contrived.


#### Curve Fitting and Extrapolation

The use of extrapolation [can be fraught with issues](https://www.xkcd.com/1007/). In this case the authors extrapolate 1-2 orders of magnitude using limited data
and, as far as I can tell, don't really justify their use of a 3rd order polynomial. That being said there is only one real zero and it does increase monotonically,
so at least in the range over which it is defined there won't be drops at higher values.

More notably the authors also don't provide any  error bars for their data. 
Below is an example of what their estimate would look like with error bars using both 
standard deviation and bootstrapping.

![Results](/images/error_bars_for_embedding_limits_polynomial.png)

The bounds around performance using 95% CI are rather large, nearly 50%. Below is table with the ranges

| dimensions | Mean | Analytic CI Lower | Analytic CI,Upper | Bootstrap CI Lower | Bootstrap CI Upper |
|------------|------|-------------------|-------------------|--------------------|--------------------|
| 512   | 509,025.71 | 332,616.85 | 685,434.56 | 90,597.41 | 720,449.71 |
| 768   | 1,698,767.93 | 1,072,650.10 | 2,324,885.76 | 225,187.76 | 2,447,487.23 |
| 1024  | 4,005,321.60 | 2,483,952.83 | 5,526,690.37 | 438,732.75 | 5,827,806.20 |
| 3072  | 107,062,547.63 | 63,930,717.43 | 150,194,377.84 | 6,684,349.74 | 158,563,955.12 |
| 4096  | 253,473,998.20 | 150,616,883.99 | 356,331,112.41 | 14,319,291.91 | 376,219,318.51 |

In short, the authors estimates could be seriously off, even assuming there are no other issues. Interestingly the bootstrap
estimate seems to have a preference for more conservative number of items. I suspect
that is due to the fact that the larger # of dims are underestimated i.e., we can see that at the tail end of the original data, the points are above the curve plotted
by the polynomial. So when bootstrapping, samples that are primarily composed of smaller values tend 
to estimate a very small number of items. I don't if this trend would hold for higher dimensions though, since the 
experimental results seem to hover around the plot line.

![Results](/images/original_polynomial_plot.png)

![Results](/images/original_polynomial_plot_closeup.png)


### Experimental Design Issues

The methodological issues are just the cherry on top. There are more fundamental issues that make even the adjusted estimates unlikely to be
correlated to real world performance. The following subsections explore the issues that, IMO, make it harder to apply this method, distort the bounds
to be more conservative, may produce variable results.

#### Difficult to Interpret Hyperparameter

The top-k value introduced to pick the threshold that don't always have a clear analog in practice. How do we even choose $k$ for a given problem? For user
facing applications this might be easier, since we can suppose people won't look at more than 2,5,10 results. What about pipelines with multiple steps?
What about RAG, which top-k is suitable then? This might seem like a nit pick, but when your entire method for estimating the minimum dimension is reliant on a hyper-parameter it is important to 
have an answer to that question of what $k$ value to pick.

#### Query/Corpus Relationship

This design choice to have as many  principled, but seems like a poor one for two reasons:
1. It makes finding these bounds difficult computationally to estimate bounds, because $|C| choose k$ grows very quickly for large values of $|C|$ and $k$
2. It's unclear how close this bound is to typical IR query document relationships, at least as captured in datasets. Is this bound close regardless of the task used, or much higher?

The number of dimensions found by this method should be higher given its likely harder to fit than other combinations of queries.
Section 5.4 and Figure 6 seem to indicate that this would be the case, although those experiments are used in a different context.

#### Variability Based on Initial Conditions

<!-- NOUR PLEASE REVIEW ME -->
Although optimization methods like ADAM are less susceptible to start conditions and getting stuck in local minima <!-- TODO: 1. verify this is true, 2. citation -->
this is a real concern. Initial conditions will impact how many dimensions are found. THe authors don't appear to account for 
this, simply running experiments once. Running multiple experiments and providing variability of number of items found would have been a good starting point.
Maybe overfitting prevents this, but not sure.
An experiment to understand how  initial conditions impact the dimension d value found would be important.


## Practical Performance

While this empirical bound is interesting, as discussed it doesn't touch on what we can expect in practice.
This section is billed as practical results, however it primarily serves to introduce a new dataset, LIMIT. The rationale is that current datasets don't have a 
high enough ratio of queries to documents and therefore can't completely test the combinations possible.
Sections 5.4 and figure 6 demonstrate that having a denser $rel$ matrix makes the task more difficult for 
recall @ 100. 

LIMIT simulates the query document relationship by creating user profiles consisting of "attributes"
and simple few word queries. Attributes are properties such as "likes peperonni pizza".
Users are limited to 50 attributes per profile and queries ask for one attribute at a time so it is a relatively
straightforward task. The attributes and subsequent profiles are generated by gemini 2.5, checked for dupes,and hypernyms (we'll circle back to this).

My critiques of this section are 2 fold
1. LIMIT is artificially, and unnecessarily, difficult
2. No practical implications of the theorem are demonstrated

### Artificial Difficulty

So the authors pitch the LIMIT dataset as an easy dataset BM-25 can do, but neural models can't.
There are three design choices that make the value of this claim questionable IMO:
1. BM-25 is used to filter for profiles that are likely to be false positives
2. Removal of hypernyms in the attributes benefits models that don't capture the semantics of words like the baseline (e.g., BM-25)
3. The domain used is unlike any the models. 5.3 aims to show this isn't the case, but 
the experiment uses a split with new attributes. So it's fundamentally a different distribution
of the dataset. It would not be shocking if the model couldn't learn p(y|x) given how different p(x) is. This isn't accounted for in their explanation.

For me the best example of this is that the colbert results are quite a bit better than single embeddings, but still lower than BM-25.
Given how Colbert is said to transfer domains better <!-- TODO: Include citation -->
I think difference can be accounted for by the three reasons listed above.

Beyond the construction of the dataset, how its evaluated is very odd. The authors aim to show 
that dimensional is a limiting factor. To this end they use a combination of different models, including about half that aren't trained with MRL.
Moreover those that are trained with MRL aren't trained to go as low as they do, <!-- todo: get a citation for this if correct -->
so the results shown are unsuprising and could be explained by experimental design issues.

### Missed Opportunities

Overall the practical section of the paper felt like a missed opportunity to actually to show the theorem 
does (or doesn't) show up in practice. The authors even have [code](https://github.com/google-deepmind/limit/blob/main/code/free_embedding_experiment.py#L363-L493) that could be used to test the $rel$ of real datasets using the free
embedding method, more on that later. IMO this would provide a more realistic 
representation of the bounds. You could even use this method on your own dataset to get an idea of the scale.
Instead the section's focus is on showing how difficult LIMIT is for neural models and justifying the design choices used to make it that way.

## Experiments I'd like to have seen

The theorem is a really good starting point for further work and I think there are a couple directions in which to take it.
This section is primarily focused on applying theory to real world and understanding the practical implications/ utility
of such a theory. I still feel like the following experiments 
don't go far enough, but they are a step in the right direction.


### Estimates of minimum dimensions over real datasets  

To estimate the minimum dimensions required over a real dataset,
 a systematic grid search across **six embedding sizes** (`64, 128, 256, 512, 1024, 2048`) for two BEIR datasets:
1. nfcorpus
2. scifact

For each dataset, the code <!-- todo: include github link -->

1. Downloads and unpacks the benchmark data.  
2. Loads documents, queries and relevance judgments (`qrels`).  
3. Converts the sparse relevance information into a binary ground truth map required by the optimizer.  
4. Calls `optimize_embeddings` with the current dimensionality together with a set of hyperparameters (learning rate, temperature, early‑stopping settings, etc.) that are pulled from `DEFAULT_EXPERIMENT_PARAMS`.  

The function returns the accuracy with which the top-k (k=2 in this case) was found.

While I was able to get this code to run, its unclear if I did something wrong or there is fundamentally an issue with this code or approach.
All the results, regardless of dimension, converged to a specific accuracy for a given dataset.

### Variability of the method over multiple runs  

To assess how sensitive the Adam based optimization is to random initialization, the script repeats every `(dataset, dim)` experiment **three times** using distinct seeds (`0`, `42`, `13044`).  
Each seed is passed directly into `optimize_embeddings`, which (presumably) seeds JAX’s PRNG and the Adam optimizer state.
Because the results of each run are stored separately in the same JSON file, you end up with a replicated set of scores for every configuration.  

Again, like above all the results are the same across seeds. However its unclear to me if this is due to an issue on my end or the code.

### Effects of Changing $rel$

The reliance on $rels$ begs some questions about how changes in $rel$ effect the dimensions required. These
changes could come from 3 places IMO
1. the task itself, i.e., shifts of $p(y|c,q)$ which is the functioning capturing the ground truth relationship.
2. new relationships, e.g., when new queries or documents are added
3. transfer between splits or domains

In other words, how does increasing the number of documents or queries drastically change the dimensions required to capture a given problem?
Moreover, how do we guarantee that the relationships in the training data and the minimum dimensionality
we determined with them are adequately high? If we need a dataset that is representative to 
accurately compute minimum dimensions that might be harder to use. Especially given that new, unseen queries
are likely a relatively frequent occurrence given the long tailed nature of queries.

<!-- text to flush out, descrube ab experu -->

Try 2, 5,10,20 fold to see how variable this is. The reason is that this will give us an idea 
for tasks where the train corpus is much smaller than real corpus how these bounds might generalize

<!--
### Could Sign Rank Approximations be used

There are methods to approximate rank, so maybe 
these could give suitable bounds. However this falls outside of the scope of the paper and 
frankly my knowledge. It seems like the rank can be computed using [SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_rank.html) or approximated with 
the [structural rank](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.structural_rank.html).

To get an idea of how variable these bounds might be an experiment was run. Across 4 orders of magnitude
of $rel$ and query size. Initialized a $C$, $Q$, $rel$ randomly. Compute both exact rank and structural rank.
Compute k+1 ranks using k-fold validation, one rank for the entire dataset and one for each "training" fold.
We then compute the distribution of the deltas and normalize with respect to the true rank value (percentage error).
This creates an error bar giving us an idea of how much the true dimensionality required is compared to just the 
one computed from the train set.
Want to answer two questions
1. If we hold out x% of the dataset, how much does the rank compare structural rank vary compared to the "true rank",in other words the minimum dimensions required to capture the entire problem
2. How close are the approximate bounds to the real bound and how much time is saved doing this.
 TODO: Create a table with results and run experiments LMAO-->



## What's next?
<!-- NOTE: THis section is a WIP, ignore for now -->


### Observing Patterns in the Real World


This has been alluded in the previous sections and experimented with to a small extent. Beyond just estimating 
the dimensions required, it would be nice to see if the pattern were reflected. Specifically, 
we would expect to see a diminution in performance around theoretical limit.
This would like present as a knee around the number of dimensions, in other words 
performance should stop increasing around that value found.

In practice this would look like performing free embedding experiments over BEIR (and/or other IR datasets)
followed by performing the tasks with different dimensions.
We would expect knees to occur near the theoretical limit in the performance chart. Maybe there is no knee or it is shifted. but that in of itself
would be an interesting finding. 
Even a negative result like this could be used to understand how optimal learned representations are in terms of efficiency.
If we can compute close bounds on dimensions, we can get insights on how efficient representations are? Are we far off the bounds?


### Computable Sign Rank or Better Estimates on it's Bounds

I'm unaware of the other uses of sign rank, but if this theory is correct a generalized and efficient version of sign rank 
that works over large sparse matrices would be very important to IR practitioners. It would allow us to quickly estimate 
rank. This falls outside of my area of expertise, but would be of immense value to practitioners given the reliability
of the free optimization method is unclear.

### Synthetic Estimates of Dimensionality

Assuming that we could reliably compute bounds and that they are accurate, could we 
use synthetic approaches to estimate the true dimension needed for a problem given 
a small sample. Lets say I only have 100 examples, would it be possible to synthetically grow $rel$ and get a good estimate of the 
true dimensionality required for a given task? Essentially figure out if can we approximate p(y|x) and if that gives us reliable estimates on $d$.

### Alternative Uses of Single Embeddings

The authors discuss alternatives having found "major limitations" of embeddings. These include sparse
representation and multi-vector representations. However given the weaknesses outlined above,
it's unclear to me that this is even true.

Assuming a fundamental limit does exist, even using the estimates presented by the paper,
a large number of documents could still be represented by a single vector. It could make sense
to subdivide a corpus into different areas or use multiple representations for a single document that are used 
under different circumstances. It is worth noting at that point, other representations might make more 
sense and it would heavily depend on the specific problem.Assuming a fundamental limit does exist, even using the estimates presented by the paper,
a large number of documents could still be represented by a single vector. It could make sense
to subdivide a corpus into different areas or have multiple representations for a single document that are used 
under different circumstances. It is worth noting at that point, other representations might make more 
sense and could be used depending on the [type of similarity](https://felixlabelle.com/2023/11/18/discussion-about-text-similarity.html) required.

### Generalizing the Theorem to Reranking

On a tangent,lets accept the premise that single embeddings have a limit on the number of documents and that it is too small for most use cases.
Lets say we decide to rely primarily on rerankers, there are still practice reasons to use single embedding models (speed, software support).
What if instead of using single vectors to get the top-k directly, I just used them as rerankers?
Instead of getting the top-k correct, all that matters in that case top-k in some larger k'.
This would open the door to  use single vectors as efficient representations instead and just 
size them so we have guarantees on k'. In a sense this already happens with some optimizations 
for multi-vector methods like plaid <!-- todo: verify and add a citation -->. In that case,
extending this theory to give bounds on what sized k' we can get the top-k results in would be interesting.

## Closing Remarks

This paper is a step in the right direction, providing theory to guide design choices when it comes  to text representation.
However it fails to provide practical insight. The proof itself seems good, but bluntly the practical findings, 
practical contributions, and conclusions of the paper are in my opinion, at best, deeply flawed.

I think this paper should be followed upon with much more thorough experiments over existing data to see if
traces of this theorem can be detected in practice. Even if we can't see a correlation between estimated bounds on dimensions and performance,
that if of itself is interesting. It points to other issues we may need to fix. Until that's done this proof is neat,
but may not have any uses for practitioners.
In short, based on this paper alone there is no clear reason to abandon single embedding vectors or increase their size endlessly.
For now keep using empirical measurements of performance over your data and evaluate the design tradeoffs between
any potential alternatives like multi-vector or sparse representations.


## References

https://news.ycombinator.com/item?id=45068986
https://arxiv.org/abs/2407.15462
https://arxiv.org/abs/2407.13218
https://arxiv.org/pdf/2205.13147