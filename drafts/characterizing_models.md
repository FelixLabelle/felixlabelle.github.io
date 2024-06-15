# Characterizing Encoders and Generalization

## What is Characterization?

In other engineering disciplines we often need to be able to answer questions when scoping out projects, figuring out the resources required,
and confirming that solutions work ideally before implementing them. While there is no replacement for real world use, we need to be able
to implement information before getting there.

For example in electrical engineering, we have many tools to predict how circuits will behave, either through equations or even simulations
of circuit behavior. We can choose component values and even predict performance before ever building a circuit. Prototypes in the form
of Breadboards, development boards, or even prototype boards are still invaluable and will allow teething issues to be identified and addressed.
However due to the cost, time, and risk involved in production it makes sense to predict performance.

While the cost and time required for software might not be as high, this still has value 

## Characterization in the Context of NLP

Currently when starting an NLP project, there are often questions I have for which there is no way (at least that I'm aware of). 

1. Which embedding space would be best for this task and domain?
2. How much data do I need?
3. What type of performance can I expect?
4. Do I have adequate coverage of examples? What type of data should I be labelling?


1. How much data do I need
2. How well can I expect a task to do
3. What type of data do I need?
4. Which embedding space is best?
5. What type of models (depth size etc) would be best for this task 
6. Estimate parameters/hps that are best suited 


While some of these questions can and have been answered independently (see Relevant Work section), I think there could be an attempt to unify this. 

As a first attempt classification tasks make more sense for a couple of reasons. 

### Classification Tasks still Matter

I think inspite of the move towards LLMs, prompting, and generative tasks there are still reasons to frame tasks as classification and finetune models for that purpose

1. Straightforward evaluation frameworks, approaches, metrics
2. Ease of labelling and error analysis compared to more open ended output spaces
3. Limited scope allows engineering tradeoffs/decisions more easily
4. More easily to explicitely design pipelines
5. (Opinion) I think explicit decomposition of a task allows for better traceability and debuggability in a pipeline.
6. A lot of tasks can be framed as limited options
7. Solving a lot of problems at once might actually be harder. I think there may be tasks that are mutually harmful in the context of current models. I don't have a strong base for this besides experiments that show that multi-objective models tend to do worse than single objective ones. It's unclear to me if scale fixed this
8. Potential for smaller models
9. Use of active learning or other frameworks that allow for efficient labelling

In practice the most downloaded model is BERT-base. It makes a good model for pretraining. 

### Characterization vs Interpretability

Characterization is not about knowing the reason behind decisions or understanding one off examples, it is about being able to predict general behaviors and trends in models.

While some research does exist, it's hard to find a unified framework or library to do all of this. There are some fundamental properties of models 

To this end I propose creating a framework with which we can

Reveals more fundamental truths about what models capture and potentially give us directions on how to improve them. 


## Idea

I want to be able to predict performance for a given task using only
1. An embedding space
2. A labelled training sample 
3. An unlabelled eval sample

This would allow me to get a better idea of performance

Ideally it could give rules of thumb as to how much and which data is needed for a given task

I want to find features that can guide us towards data that needs to be labelled for a given embedding space. I'd like to have an embedding space independent method, but it is what it is.

Solution si to 

1. Train a bunch of linear models on varying subsets of training data
	1. Select datasets (GLUE for Pilot study
	2. Create a large number of embeddings 
	3. Create subsets of training and eval data
	4. Create different metrics or groupings of dataset
	5. Take different levels of data 
2. Train a metamodel that predicts performance
	1. Determine what features might capture relationships  
	2. 
	
### Relevant work

1. Transfer learning between tasks (Intermediate learning)
a. https://arxiv.org/abs/2005.00628
b. https://aclanthology.org/P19-1439/ 
c. https://arxiv.org/pdf/2005.00770.pdf  (defines task embeddings, seems most relevant) 
 
2. Probing
a. https://aclanthology.org/2022.emnlp-main.793.pdf 
b. https://arxiv.org/pdf/2211.06420.pdf 
c. https://arxiv.org/pdf/2202.12801.pdf
 
3. Few sample finetuning
a. https://arxiv.org/pdf/2006.05987.pdf 
 
4. Finetuning ideas
a. https://aclanthology.org/2023.findings-acl.889.pdf

Need to add:
Peter Anderson paper
Metalearning literature
Optimal Experiment literature

## Experiment

WIP, still doing analysis

Experiment run, still picking out features and seeing if it works


## Results

## Closing Remarks

