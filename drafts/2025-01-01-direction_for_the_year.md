# Improving the NLP Tool Kit: Characterization
<!-- TODO: Read through and add citations -->
IMO the last couple of years have seen a shift in NLP research and products, moving from solving specific problems to more general solutions.
Five or six years ago, papers often proposed a single dataset or architecture aimed at improving performance on a single task or class of tasks such tagging.
Some early work on better representations did exist like word embeddings, contextual embeddings, or BERT models, but most of the datasets to evaluate this were smaller
in scope (think GLUE). <!-- TODO: Add citations here -->
Similarly, the few successful NLP tools in deployment at the time had a single function, such as search and translation (think search engines, translation tools) and 
required quite a bit of work (engineering, data, cheesing) to get working properly.

With the advent of GPT models, it feels like text generation has taken center stage. Gone are the days of gathering data and training a specialized model for some scoped down task.
Models can now generate open ended outputs and therefore answer and be evaluated on a large number of tasks.
Moreover, the expectations of what problems models should be able to solve has changed as well.
Implicitly tools are increasingly expected to excel at something beyond a single purpose; terms like "AGI" are [thrown](https://arxiv.org/abs/2303.12712) [around](https://openai.com/charter/).

These advances are interesting, but sometimes it feels as though a step was skipped. The field has moved directly from solving individual problems to attempting to solve all problems.
However even these improved models don't get perfect results. As an engineer there is still a need for task specific solutions, whether it be due to constraints like minimum performance, time, cost, and hardware.
When scoping out projects, we need the ability to estimate resource requirements and confirm solution viability before implementation. 
 In spite of recent advances IMO there is a lack if tools to help make design decisions for 
specific tasks. There's no straightforward way to build a maintainable, debuggable model-based classifiers quickly and efficiently or predict how a solution will perform in the field.
Instead of striving to solve problems "in general," we should aim to solve general problems about ML effectively. Understanding how models perform in certain settings is
just as interesting if not more so than a model that can do everything in my opinion.

## Overarching Goal: Characterization

One way to scope projects is using measures that are predictive of downstream performance/behavior of the approach being evaluated (specifically the underlying model).
This is analogous to **characterization** in fields like material sciences; creating measurable properties and measures that help determine a tool's technical capabilities. 
While this is not necessarily a novel idea even for NLP, as some instances already exist such 
as scaling laws, the work I'm aware of is disparate and characterization is not a field in it's own right.

### Examples of Characterization in Other Engineering Fields
<!-- TODO: Unchatgpt this section -->
Characterization is used in other engineering fields. for example:  

1. **Circuit Design:**  
   In circuit design, characterization involves using measurable properties like resistance, capacitance, and inductance to predict how components will function in a circuit. These properties allow engineers to:  
   - Design filters or amplifiers without needing to build physical prototypes.  
   - Understand voltages and currents at different points in a circuit, enabling prediction of how the system behaves under various conditions.  
2. **Material Sciences:**  
   Engineers and scientists use properties like tensile strength, elasticity, and thermal conductivity to characterize materials. These measurements allow predictions about a material's suitability for specific applications, such as construction, manufacturing, or electronics.  
3. **Mechanical Engineering:**  
   In mechanical systems, properties like torque, friction, and efficiency are characterized to design machinery. For instance, predicting how a gearbox will perform in different environments can prevent failures and optimize energy use.  

These examples highlight how measurable properties can save time and resources by providing actionable insights during the design phase, ensuring that a solution will meet desired outcomes without excessive trial and error.  

### Characterization in NLP

Essentially we want to turn a given approach (e.g., an embeddings) into a bunch of numbers and assess viability for a given downstream task (i.e., answer specific questions).
To simplify this task I want to start by focusing on classification tasks using a linear layer over a single embedding. This specific approach was chosen because:
1. classification is still a very common task
2. characterization of embeddings gives you fundamental properties

With this framing in mind, first step is to identify questions we might want to answer. 
There are many questions for which no clear answers exist (at least to my knowledge):
<!-- TODO: reorganize the following list and match it up point 2 of the road map -->
1.  Which embedding space would be best for this task?
2. Impacts of domain invariance within 	my data
3. How data length variations impact performance for a proposed
4. lengths of data (at least max length)
5. Do I have adequate coverage of examples? What type of data should I be labeling? Do I have enough data?

Second step is to find measures that are useful in answering these questions (see the road map below).
Once these measures are available rather than try a few attempts based on intuition we could down select using appropriate measures.
Moreover these measures
could be used to create better representations for a given problem. Lets say we know a given embedding can only represent 
documents of length 512 for IR, we could do something like fuse multiple embeddings from different models or layers.

#### Assumptions

This vein of work was chosen with several assumptions in mind. I don't necessarily have citations
or justifications for any of these statements nor do I plan on providing any.

1. **Made-to-specification solutions are more efficient** (resource-, time-, and cost-wise) than open-ended approaches for well-defined problems. This doesn’t mean few-shot prompting is bad, but specialized systems can simplify assumptions to reduce costs or even improve performance.
2. **Most engineering problems can be well-defined. If they are more complicated they can solved through decomposition.** This is based on my intuition and observations at work, where solutions often require modular building blocks.
3. **Classification is a good focus for Characterization**
   - It is more understandable. Failure rates can be calculated, resolution strategies tailored, and bugs isolated more easily than with black-box models.
   - It is easier to evaluate. Open-ended generation has a vast space of valid outputs, making it inherently harder to evaluate. Even within that space, evaluations often focus on specific facets that can be reframed as classification tasks (e.g., Factuality).
   - It is often sufficient for the task at hand. For example, generating an explanatory text about why a document is poorly written may overlook key facets of "goodness," especially with specialized data where such definitions are precise.
   - It is versatile and can serve as filters, information enhancers, or tools for finding similar or "confusing" items.
4. **There exists measures that can be used to capture downstream performance across a variety of settings** That's not to say they will be easy to find, just that I think they do exist.
5. **Being a non (m|b|tr)illion-dollar entity focusing on simpler tasks and characterization is a good way to compete/create useful models**

### Relevant Work

This is by no means a comprehensive list, if you see a missed reference feel free to comment or contact me.
Part of the difficulty with the task of collecting a list of relevant works is that while it
is possible to find small clusters of related works, but I haven't yet
found a well flushed out field.

1. Transfer learning between tasks, representation learning
	1. https://arxiv.org/abs/2005.00628
	2. https://aclanthology.org/P19-1439/ 
	3. https://arxiv.org/pdf/2005.00770.pdf
2. Probing
	1. https://aclanthology.org/2022.emnlp-main.793.pdf 
	2. https://arxiv.org/pdf/2211.06420.pdf 
	3. https://arxiv.org/pdf/2202.12801.pdf
3. Few sample finetuning
	1. https://arxiv.org/pdf/2006.05987.pdf 
4. Finetuning ideas
	1. https://aclanthology.org/2023.findings-acl.889.pdf
5. Predicting performance
	1. https://aclanthology.org/P18-2072/
6. Meta learning literature (honestly I'm less familiar with this area of research. Need to read more here)
7. Optimal Experiment literature (honestly I'm less familiar with this area of research. Need to read more here)

## Road Map for 2025

Defining and validating the measures is the road map. First step is to build a way of validating these measures.

1. **Create a large classification-only dataset** to enable meaningful experiments. This won't be an effort from scratch, rather aiming to curate existing benchmarks and datasets. I want to do this rather than use a benchmark like GLUE, SuperGLUE or BigBENCH because 1) it allows for a deeper understanding of the data present 2) there are potentially [questions of quality with some existing datasets](https://cs.nyu.edu/~davise/Benchmarks/BIG-bench.html).
2. **Conduct early experiments** to identify measurable properties that correspond to downstream performance, even in limited settings. First attempt will likely be on carrying capacity (what length of text a given embedding can capture).
3. **Develop techniques informed by these measures** to improve performance and confirm the validity of the identified properties.
   - ModernBERT and other encoder only models
   - Potential directions:
     - Mixture of Experts (MoE)
     - Improved tokenizers
     - Different objectives
     - UNet-like architecture to increase carrying capacity
   - Non-architectural directions:
     - Instruction-tuning smaller models
4. ** Expand to other problem types** such as IR, sequence tagging.

## Closing Remarks

Characterization is an attempt to make NLP projects more predictable, efficient, and effective.
By focusing on measurable properties and how those translate to technical capabilities, the gap between abstract model performance and real-world engineering needs can be bridged.

The first step is building a large validation dataset to serve as a foundation for exploring these ideas.
This dataset will allow experimentation aimed at uncovering the relationships between data, model choices, and downstream performance.
The hope is that this groundwork will help create practical tools and frameworks that engineers can rely on when designing NLP systems.