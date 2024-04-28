# "Nudging" Active Learning

## What is Active Learning?

If you are familiar with active learning, feel free to skip this section. If not, I will introduce basic concepts.

Active learning is a form of data labeling where the model is at the heart of the process. Unlike traditional labeling, where 
data is sampled ahead of time, active labeling trains the model in parallel. The idea is that this is more data efficient. The model being trained, or another model,
is used to select new samples to label. 

The method used to select new samples is known as the *acquisition function*. There are many
acquisition functions, but we will only discuss one family of methods in this blog:
1. Confusion based methods: Use model confidence (class probabilities) to determine which items the model is unsure about. There are different ways of doing this, like entropy which is what I typically use. Less certain
(higher entropy) outputs are prioritized for labeling

There are drawbacks to this method, but when I bench marked it using some binary data (including proprietary sets from work) it gave the best convergence and performance for the 100-200 sample range (which is realistically what I'd like to label or can get labeled by others).

If you are looking for
a more in depth resource on Active Learning I enjoyed [Robert (Munro) Monarch's Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning). There
are other good resources, but this book gives a good foundation IMO

## Why Active Learning?

At work I use active labeling to create models for upstream tasks like data filtering. Active labeling, specifically use of a confusion based acquisition function, 
appears to give relatively good results over some benchmarks with relatively few labels, typically 100-200 samples for binary tasks I've tried it with such as this [anti-lgbt bullying dataset](https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts).

The process is simple, I first randomly sample 10-20 samples, then afterwards use an confusion acquisition function, specifically entropy. High entropy (less spiky probabilities) are prioritized
and sampled first.

## The Nudge Method

There are some limits to active learning, in particular when it comes to labeling datasets with minority classes. Recently I trained a model to recognize "poorly formatted text" using explicit features
(punctuation/char ratios). The task in question was over 50k+ examples, but the poor examples
number in the 200-500 range; <= 0.1% of the dataset. Currently the method I use for active labeling involves a random initialization
period, i.e., draw n samples randomly, training the model, preparing the acquisition function, and then using another acquisition function for the rest, retraining the model at intervals t (t=20 in my case).

If examples of a poorly written sentence are rare, the chances of sampling a bad example are very low. This leads to an issue for 
the confusion-based acquisition function. Without all of the labels its unable to predict missing class(es). In my first run training this model, the model went through 50 samples without
pulling a single bad one.

This cold start issue means you need to luck out and find a relevant sample. While there is literature on [cold starts](https://aclanthology.org/2020.emnlp-main.637.pdf) and [minority labels](https://arxiv.org/pdf/2201.10227) in active learning,
the methods seemed a bit more complicated than what I needed.

What I came up with is the "nudge method". Transform the binary problem into a ternary one by adding in a pseudo label (the nudge label). The nudge label is used to tag instances that, while not what you are looking for, are close
enough. In my case I was looking for texts with odd formatting, so any text that was somewhat formatted would get a nudge label. At the end you replace all the nudge labels with the "good" label.

Subjectively this worked. First off, after the initial random labeling, with one nudge example, the model flagged several bad examples. So rather than go 50 labels dry, I found bad examples within 10. Secondly sampling through tagged examples,
the model seemed to do relatively well (sorry no numbers, but out of a sample of 20 good and 20 bad everything looked correct).

## Caveats

My use case was very simple and I think that's why this method worked

1. I have a feeling the success of this method depends on the feature space. In my case the features have a pretty clear and likely monotonic relationship to poor examples (individually). Would a higher dimensional input, like TF-IDF or neural embeddings have been as easy to work with? Not so sure
2. The poor examples are relatively easy to flag and there were "intermediate" samples (i.e., texts with more formatting). This method wouldn't have worked as well if there weren't "bridge" items ("bad-adjacent").
3. The model used was a low-depth (1) decision tree. It's rather simple and likely implements something similar to thresholding. Not sure if this played a role, I need to do more experiments.
4. Binary vs multi-class. While there are ways to generalize this (use a shared nudge label or multiple nudge labels), I'm not sure how they would fare

I don't have any examples, code or data since this was done for work. so my apologies for that.

## Conclusion

Confusion based acquisition functions can have a cold start issue when labeling a minority class. When doing a simple, binary class classifier it is possible to introduce a pseudo-class to tag "bad adjacent"
examples. This encourages the model to pull out examples it is confused about that are "bad-adjacent", including "bad" examples. For me this worked well by reducing the time required to find a "bad" example.