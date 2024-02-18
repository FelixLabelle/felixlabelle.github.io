# Characterizing Encoders and Generalization

## What is Characterization?

In other engineering disciplines, like electrical engineering, there is a need to be able to simulate. While there is no replacement
for implementing circuits in real life, there is a practical value to being able to predict how circuits will behave before
deploying them out in the real world. Currently NLP is lacking any way to do this AFAICT.


## Characterization in the Context of NLP

While some research does exist, it's hard to find a unified framework or library to do all of this. There are some fundamental properties of models 

To this end I propose creating a framework with which we can

Reveals more fundamental truths about what models capture and potentially give us directions on how to improve them. 

### Classification Tasks still Matter


### How to Measure Characterization

microstructure and physical properties are probed, measured and determined using a variety of analytical methods, techniques and tools.

While Encoders are "demode" (out of fashion), I think there is value in using them in this context

1. Classification and sequence labelling still have value in the NLP toolkit. 
	1. A lot of problems can be framed as classification and sequence labeling
	2. THese tasks are measurable and we have a relatively well understood framework to do measure success and do error analysis
	3. We can gather data for a specific task that may be hard to describe or on OOD data that a pretrained LLM with prompting may not do as well
2. Encoders like BERT or RoBERTA are rather good for this task:
	1. In production the latest models are slow and cannot be quickly inferenced on CPU. This means they can be used for servers or preprocessing workflows easily
	2. Encoders can and give very good performance
	3. Encoder are easy and relatively cheap to finetune. They are available for multiple parts of hardware
3. Encoders aren't getting a lot of love and updates
	1. There has been a limited amount of Encoder only models since 2021 and the state of the field has been limited
	2. There have been a lot of improvements both in encoder only models and transformers overall. Our understanding 
	

For these reasons I want to improve models 
1. How we can improve ENcoder only models based on research from the last few years

In a lot of engineering disciplines we can charachterize how materials or systems will perform on a new task. As far as I know,
this is not something that's common practice in NLP.

In future work I want to go beyond that and understand how we can improve these models beyond benchmarks.
I'd like to be able to characterize models and predict how much data we would need to train it and over time reduce that
so these models can be used to do few shot with known guarantees on performance

## Study Approach

There are three parts
1. Train and benchmark a new model called NeoBerta on larger datasets. Compare to RoBERTA and BERT etc..
2. Develop a reliable method to predict performance on a given dataset with T training data for a given metric M
3. Use 1 and 2 to develop a sample efficient way of transfer learning

## Parts of NeoBERTA training

* Neoberta
	* Tokenizer
	* Masking training
	* CLS training (nsp prediction)
* Neoberta downstream finetuning
	
## Questions Characterization should be able to answer

1. How much data do I need
2. How well can I expect a task to do
3. What type of data do I need?
4. Which embedding space is best?
5. What type of models (depth size etc) would be best for this task 
6. Estimate parameters/hps that are best suited 


## Pilot study for characterization

1. Train a bunch of linear models on varying subsets of training data
	1. Select datasets (GLUE for Pilot study
	2. Create a large number of embeddings 
	3. Create subsets of training and eval data
	4. Create different metrics or groupings of dataset
	5. Take different levels of data 
2. Train a metamodel that predicts performance
	1. Determine what features might capture relationships  
	2. 
	
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
## References

<!-- https://en.wikipedia.org/wiki/Neural_scaling_law -->
https://huggingface.co/abarbosa/c4-aristo-roberta-large
https://huggingface.co/joelniklaus/legal-english-roberta-base
https://discuss.huggingface.co/t/robit-pretrain-roberta-base-from-scratch-in-italian/7564
https://arxiv.org/pdf/2305.13169.pdf
https://huggingface.co/antoinelouis/netbert
https://huggingface.co/NepBERTa/NepBERTa

### Training an LLM
https://huggingface.co/blog/how-to-train
https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/01_how_to_train.ipynb#scrollTo=YpvnFFmZJD-N
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb

#### Architectures to look at
https://huggingface.co/docs/transformers/model_doc/roformer
https://huggingface.co/docs/transformers/model_doc/roberta
https://huggingface.co/docs/transformers/model_doc/albert
https://huggingface.co/docs/transformers/model_doc/bert

#### Tokenizers?
https://huggingface.co/docs/tokenizers/v0.13.4.rc2/en/api/models#models
https://github.com/alasdairforsythe/tokenmonster
https://huggingface.co/alasdairforsythe/tokenmonster
https://huggingface.co/NovelAI/nerdstash-tokenizer-v2

### Datasets

#### Tooling

https://github.com/p-lambda/dsir
https://arxiv.org/abs/2212.10440
file:///C:/Users/flxla/Downloads/7307-Article%20Text-10537-1-10-20200601.pdf

#### Pretraining

#### Downstream tasks

##### Classification

##### Sequence Labelling


### Characterization (Ask Peter)
task_vec + number_samples + cross_fold + kv_divergence_between_datasets + sample_size + downstream_size => mean +- range 
quantile regression
https://en.wikipedia.org/wiki/Quantile_regression
https://arxiv.org/abs/1902.03545


### Model selection
Find model that does best sample wise