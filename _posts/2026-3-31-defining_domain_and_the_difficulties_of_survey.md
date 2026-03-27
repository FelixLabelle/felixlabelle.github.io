# Impact of Different Settings on Generation Length


Research questions

1. What decoding has an impact on generation length
	a. temperature
	b. top-k + temperature
	c. top-p + temperature
	d. repetition_penalty + temperature
	e. reasoning settings
2. What has an impact on generation time
	a. 

Facets to test 
1. different tasks (need to inventory datasets)
	a. https://huggingface.co/datasets/Muennighoff/natural-instructions/viewer/default/train
2. different domains (define using can human id domain paper, use L1 for subjects)
3. different models
	a. size
		i. 1B
		ii. 3B
		iii. 8B
		iv. 14B
		v. 30B
	b. density
		i. sparse
		ii. dense
	c. model family
		i. llama 
		ii. qwen 
		iii. olmo??
	d. age
		i. 2023
		ii. 2024
		iii. 2025
	e. training degree
		i. chat
		ii. instruct
		iii. raw
	f. reasoning vs non-reasoning
4. different model configuration
	a. kv ?
	b.  underlying stack used 
		i. llama.cpp
		ii. vllm
	c. underlying tool
		i. vulkan
		ii. AMD middleware (forget name)
5. decoding settings:
	a. temperature
	b. sampling type
	c. top_p
	d. top_k

	
https://aclanthology.org/2024.lrec-main.1361.pdf

metrics
1. time for Generation
2. num tokens
	a. total
	b. delta to 0 temperature
	c. refusal vs not refusal
3. summary stats over num tokens
4. Performance 
	a. (effect on rouge/blue) between gt
	b. judge (what model should I use, etc..)
	c. human eval
5. Track refusals
6. Track degenrate outputs
	a. repetiveness of n-grams
	b. other quality metrics?


Open questions
1. Amount of data (sampling strat)
2. Realistically how many experiments I can run
3. How to divide and conquer  experiments

<!---


I need your help creating Python code to run an experiment. I'll describe the experiment,


Research Question:
Does temperature effect sample length. Specifically does the temperature
increase it, by decreasing the chance of picking the SOS token

Hypothesis:
Across different models, we will see a correlation between
higher temperature and increase an token size

Experiment:
Using huggingface, we will generate stories and measure the length in tokens.
There are a couple steps for that

1. Sample 200 prompts from this datase (euclaise/writingprompts, under the column "prompt").  Make sure this is the same everytime
2. Generate stories for each prompt over different combinations of models and temperatures
	a. models 
		#Reference model
		Qwen/Qwen3-8B

		# Different generations
		Qwen/Qwen-7B-Chat
		Qwen/Qwen2-7B-Instruct
		Qwen/Qwen2.5-7B-Instruct

		# Different sizes
		Qwen/Qwen3-1.7B
		Qwen/Qwen3-4B

		# Different model families
		meta-llama/Llama-3.1-8B-Instruct
		allenai/OLMo-2-1124-7B-Instruct
	b. temperatures [0,0.1,0.2,..,1.0]
	c. different starting seeds for the generation [42,10343,202]
3. Measure the length in number of tokens for each story
4. Write the results to a csv file (save it as we progress through the dataset). The data should be
	a. model name
	b. prompt_id (just a number from 1-200)
	c. temperature
	d. text generated
	e. number of tokens 


Once that's done, we'll need to analyze the results

Can you 
1. calculate the delta in token length between temperatures between 0 and all temperatures above it
2. measure the correlation using spearman's rank between the delta length and temperatures
3. create plots for each model showing the generation

Expected results

A correlation between the length and temperature thinking [0.3-0.5]
--->