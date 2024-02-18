Keyword Summarization/extraction

Use LLMs to extract keywords for trends and use the KW to track. Cheaper to do it this way
Easier to interpret

Using models to generate programs to get answers

Using modle to generate regex etc..

Considerations
	Does the model produce extra non sense
	Does Safety cause issues
	
What's metagen?

Unifying different task types with one model
	Looks like there can be a gain
		Data type (example seen uses images)
		Data required?
		Preprocessing required (class balancing, sampling strats, etc..)?
	Use a VERY LARGE training dataset
	Use a feedback system to improve
		Can train models and then use them in next iteration
	
	
They are much bigger scaled than we are, but technique wise they are using simplistic stuff and hybrid techniques as well
	Established internal datasets and tasks
	Compute and infrastructure
	
Size of models is still not as massive as one would think:
	143M params (500 million training examples)
	
Also a generation behind like we are
	CNN vs ViT

Training models from scratch and full models (do not look to be using training tricks)

XRI


Ensembling prompt outputs

Different prompt task framings
1. Comparative prompt outputting, predict two labels given two inputs at the same time
2. Independent prompt outputs (do each label Independently)

Ensembling using these different prompt framings and average p(y|x)

ICCL with LLM