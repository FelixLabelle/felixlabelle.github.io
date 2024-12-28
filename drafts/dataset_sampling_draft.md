<!--
the scope of this post. I will only focus on improving feature hash representations for ngrams.
That leaves three questions:
1. How do I measure similarity (in a way that is meaningful for pretraining and downstream tasks)?
2. How do I select data?
3. How can I do this efficiently?


## Related Work

I'm not the first person to want to subselect data. This list isn't meant to be comprehensive, rather its just the papers I'm aware of as of writing this:

DSIR
Fasttext Domain Separation Model. Serves as a baseline
	Need to test this
	I think I would get comparable speeds to hashing, maybe higher 


## The Approach

So we just discussed other approaches, why did I use yet another one?

For context I filtered FineWeb, which has 23 billion documents. This imposes 2 constraints
1. Streaming; I can't realistically download the entire dataset, it's 100+ TB
2. Speed; I need to run at 10k+ instances per second

1. I tried using DSIR, chose not to use it for several reasons:
	1. Streaming data isn't really an option (OOMs, stream crashes (not DSIRs fault), and arbtirary bugs I haven't really understood that seem like DSIR issues)
	2. Speed, I couldn't match the speed in their benchmark. They claim to be averaging ~600 items per second per core, I was seeing speeds to closer to ~100 items per core
	4. It's a research project and doesn't appear maintained
	5. The downstream performance DSIR and other approaches is minimal and given the limited testing doesn't justify the tech debt or using or reimplementing
2. FastText is fast, simple, but no longer maintained. I don't want code that ages being an integral part of my pipeline

I tried originally using DSIR, but it didn't work on the scale of data I needed it to. Their FastText baseline is significantly faster and better

However I didn't want to use FastText since it was depreciated.

These papers both use Hashing in some way, so I decided to use that. Works pretty well 

For 1. the naive answer is n-gram occurences. If a text has similar n-grams to the target domain we can assume it has similar content
2. is a bit trickier. N-grams can't be easily worked with due to the shear number of them. However there is a solution, use of feature hashing

At work this approach has proven to work relatively well and turns up a fair number of relevant documents

Filtering a large web corpus creates the need for an ultra-light binary classifier to separate different text domains.
A corpus such as HuggingFace's FineWeb contains 23 Billion documents.. For context, even a classifier that runs 10k iter per 
seconds would still take 27 days to filter the entire datasets.

This requires much lighter models. When getting to these speeds larger neural networks are out of the question. This makes resort
to simple word count models. In a similar vein 

--->