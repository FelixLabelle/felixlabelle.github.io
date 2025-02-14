# Work In Progress: LLMs for ETL

<!-- TODO: Remove excessive use of personal pronouns -->
Processing data can require a fair amount of manual work; i.e., it can be a time consuming and tedious process. 
This begs the question, can data cleaning be accelerated or even automated.
While it seems unlikely current LLMs could automate the entire process,
there may well be a role for them in current ETL pipelines. Specifically in tasks where a model
could output something a human or written rules could observe, interpret, and correct as need be.
In other words rather than have a model make decisions, have it participate in a process that has verifiable inputs.

I'm currently building a [characterization dataset](https://felixlabelle.com/2025/01/01/direction_for_the_year.html).
It is a larger, classification only, benchmark built on a large number of existing datasets (~300). To this end
every dataset needs to have each task extracted and identified. Moreover each task needs to 
have a shared naming convention for the input columns and outputs. Extracting tasks and mappings from existing column names to the standard format could be done by hand.
However with ~300 datasets alone manual mappings would be cost prohibitive.

This blog will explore LLMs for ETL in the context of this problem.
The following sections present the problem, methodology used, and results.
Specifically with the goal of understanding how well LLMs perform for this task and if they are worth using for (parts of) ETL.

## Context

The use case creating a massive dataset with a fixed format from existing datasets. 
While there do exist supersets of classification tasks, none are quite as large as I'd like.
Specifically in terms of diversity of text data and tasks (for a given text, having multiple labels).
This process is comprised of 3 steps:
1. Downloading datasets
2. Removing datasets that aren't a good fit (too little data, can't be loaded, etc..)
3. Extracting tasks from datasets and mapping the original columns to a standard format

<!-- TODO: Explain dataset + config -->
### 1. Downloading Datasets
215 datasets were downloaded from HuggingFace, specifically the top 150 (5 pages, each with 30 records) based on likes, downloads, trends on 2/1/2025.
There is overlap between these three groups and errors downloading, so in the end there were only 215 datasets. Note that I stripped away the group that generated the dataset,
in hind sight this was a mistake. HuggingFace doesn't enforce unique names for datasets, just user & dataset. I will add back
user info in a future version.

<details>
<summary>Huggingface Datasets Used (Dataset Name Only)</summary>
abstraction_and_reasoning_corpus
abstract_narrative_understanding
anachronisms
analogical_similarity
analytic_entailment
arithmetic
ascii_word_recognition
authorship_verification
auto_categorization
auto_debugging
bbq_lite
bbq_lite_json
bias_from_probabilities
boolean_expressions
bridging_anaphora_resolution_barqa
causal_judgment
cause_and_effect
checkmate_in_one
chess_state_tracking
chinese_remainder_theorem
cifar10_classification
codenames
code_line_description
color
com2sense
common_morpheme
conceptual_combinations
conlang_translation
contextual_parametric_knowledge_conflicts
context_definition_alignment
convinceme
coqa_conversational_question_answering
crash_blossom
crass_ai
cryobiology_spanish
cryptonite
cs_algorithms
cycled_letters
dark_humor_detection
date_understanding
disambiguation_qa
discourse_marker_prediction
disfl_qa
diverse_social_bias
dyck_languages
dynamic_counting
elementary_math_qa
emojis_emotion_prediction
emoji_movie
empirical_judgments
english_proverbs
english_russian_proverbs
entailed_polarity
entailed_polarity_hindi
epistemic_reasoning
evaluating_information_essentiality
factuality_of_summary
fact_checker
fantasy_reasoning
few_shot_nlg
figure_of_speech_detection
forecasting_subquestions
formal_fallacies_syllogisms_negation
gem
gender_inclusive_sentences_german
gender_sensitivity_chinese
gender_sensitivity_english
general_knowledge
geometric_shapes
goal_step_wikihow
gre_reading_comprehension
hhh_alignment
high_low_game
hindi_question_answering
hindu_knowledge
hinglish_toxicity
human_organs_senses
hyperbaton
identify_math_theorems
identify_odd_metaphor
implicatures
implicit_relations
indic_cause_and_effect
intent_recognition
international_phonetic_alphabet_nli
international_phonetic_alphabet_transliterate
intersect_geometry
irony_identification
kanji_ascii
kannada
key_value_maps
known_unknowns
language_games
language_identification
linguistics_puzzles
linguistic_mappings
list_functions
logical_args
logical_deduction
logical_fallacy_detection
logical_sequence
logic_grid_puzzle
long_context_integration
mathematical_induction
matrixshapes
medical_questions_russian
metaphor_boolean
metaphor_understanding
minute_mysteries_qa
misconceptions
misconceptions_russian
mnist_ascii
modified_arithmetic
moral_permissibility
movie_dialog_same_or_different
movie_recommendation
multiemo
multistep_arithmetic
mult_data_wrangling
muslim_violence_bias
natural_instructions
navigate
nonsense_words_grammar
novel_concepts
object_counting
odd_one_out
operators
paragraph_segmentation
parsinlu_qa
parsinlu_reading_comprehension
penguins_in_a_table
periodic_elements
persian_idioms
phrase_relatedness
physical_intuition
physics
physics_questions
play_dialog_same_or_different
polish_sequence_labeling
presuppositions_as_nli
program_synthesis
protein_interacting_sites
python_programming_challenge
qa_wikidata
question_answer_creation
question_selection
real_or_fake_text
reasoning_about_colored_objects
repeat_copy_logic
rephrase
results
rhyming
riddle_sense
roots_optimization_and_games
ruin_names
salient_translation_error_detection
scientific_press_release
self_awareness
self_evaluation_courtroom
self_evaluation_tutoring
semantic_parsing_in_context_sparc
semantic_parsing_spider
sentence_ambiguity
similarities_abstraction
simple_arithmetic
simple_arithmetic_json
simple_arithmetic_json_multiple_choice
simple_arithmetic_json_subtasks
simple_arithmetic_multiple_targets_json
simple_ethical_questions
simple_text_editing
simp_turing_concept
snarks
social_iqa
social_support
spelling_bee
sports_understanding
squad_shifts
strange_stories
strategyqa
subject_verb_agreement
sudoku
sufficient_information
suicide_risk
swahili_english_proverbs
swedish_to_german_proverbs
symbol_interpretation
taboo
talkdown
tellmewhy
temporal_sequences
tense
text_navigation_game
timedial
topical_chat
tracking_shuffled_objects
training_on_test_set
truthful_qa
twenty_questions
understanding_fables
undo_permutation
unit_conversion
unit_interpretation
unnatural_in_context_learning
unqover
vitaminc_fact_verification
web_of_lies
what_is_the_tao
which_wiki_edit
winowhy
wino_x_german
word_problems_on_sets_and_graphs
word_sorting
word_unscrambling
yes_no_black_white
</details>

### 2. Removing Irrelevant Datasets

Huggingface datasets are comprised of "configs" and "splits". Configs are different groupings of data,
splits are divisions in those groups of data. To properly extract datasets every dataset is gone through
and has each config checked.Dataset config pairs that were not loadable, did not have a train split, or were too small (< 1000 train samples) were removed.

### 3. Extracting Tasks and Mapping Original Columns (LLMs for ETL)

The remaining dataset & config pairs were run through a prompt to extract the tasks within that dataset.
A task in this context is a series of one or more text inputs, one output, and a task type.
The four task types considered were
1. Regression:  One or more texts are turned into a single number (e.g., similarity tasks).
2. Text Generation:  One or more texts are used to generate output.
3. Sequence Labeling:  One text has parts labeled or extracted.
4. Classification:  One or more texts are turned into a single label (e.g., sentiment analysis).

<!-- TODO: Discuss verifiability of problem and how it makes this problem a good candidate -->
In addition to classifying the task, relevant columns for that task are pulled out and mapped
to default names. The default names are 
- text_[0-9]+ for all inputs
- label for the output
- task type for the task type

Task extraction and mapping has interesting properties,
in particular that the output can be partially validated through use of rules. The rules implemented were
1) text_1 must always be present
2) label must always be present
3) all columns mapped to must be in the original dataset
4) there can't be any columns that don't find the regex (text_n[0-9]+|label|task_type)

This means we can discard or even limit the outputs the model can generate. Only the former was performed. This was done using a Pydantic data model:

<details>
<summary>Pydantic Data Model for a Task</summary>
class Task(BaseModel):
    allowed_values: Set[str]
    data: Dict[str, Any]
    
    @field_validator("data")
    def validate_dict(cls, value: Dict[str, Any],values) -> Dict[str, Any]:
        seen_keys = set()
        allowed_values = values.data['allowed_values']
        key_pattern = re.compile(r'^(text_\d+|label|index|task_type)$')  # Custom naming convention
        required_keys = {"text_1", "label", "task_type"}

        missing_keys = required_keys - set(value.keys())
        if missing_keys:
            raise ValueError(f"Keys were missing from the requirements {missing_keys}")
        for key, val in value.items():
            if key in seen_keys:
                raise ValueError(f"Duplicate key found: {key}")
            seen_keys.add(key)
            
            if not key_pattern.match(key):
                raise ValueError(f"Invalid key naming convention: {key}")
            
            if key =="task_type":
                # TODO: Add check
                pass
            elif val not in allowed_values:
                raise ValueError(f"Invalid value: {val} for key {key}")
            
        return value
</details>

The end result is list of tasks and standardized format. Each key represents the standardized output present.
The values are the corresponding input dataset columns. Here is an example:

<details>
<summary> Resulting Output </summary>
{% raw %}
{"label": "label", "task_type": "Classification", "text_1": "sentence1", "text_2": "sentence2"}
{% endraw %}
</details>


A prompt is used to extract the tasks and generate the mappings. This would otherwise be done by hand.
The prompt was tuned over 4 smaller pilots (5,10,20,20) to find potential issues before running 
over the entire corpus. Some lessons learned here include limiting use of Markdown, however this may well 
be model specific. In the end, the prompt used was:

<details>
<summary>V5 Prompt Used</summary>
{% raw %}
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

You need to extract all tasks from datasets and transform the fields to a common format. Given a list of keys and their types, you must create a list of mappings. Each dataset could contain multiple tasks, each task will have mappings and a task type. Each mapping will include the index (if available), text(s) that relate to the labels, and the labels itself.
You will be given a list of fields and their corresponding data types for each dataset.Datasets potentially containing multiple tasks with varying numbers of text fields and labels.

The output will be a list of JSON objects where each object has the following structure:
  - A field for each text field and the corresponding input field
  - A field for each label and the corresponding input field
  - An "index" field if applicable.
  - A "task_type" field to identify the type of the task

Your task is to 
1. **Extract Tasks:** From the dataset, identify the tasks. A task could span multiple rows or columns.
2. **Convert Fields:** For each task, convert the text and label fields according to the provided JSON structure.
3. **Handle Different Field Configurations:** Be prepared to handle different numbers of text fields and labels for a single task.

Here are some rules to follow
- Always return the output as a JSON array, even if there is only one task
- The "index" field is optional and may not be applicable for all datasets.
- The "task_type" field categorizes the type of task to make the data more useful, tasks can be one of four types:
1. Regression:  One or more texts are turned into a single number (e.g., similarity tasks).
2. Text Generation:  One or more texts are used to generate an output text
3. Sequence Labeling:  One text has parts labeled or extracted.
4. Classification:  One or more texts are turned into a single label (e.g., sentiment analysis).
- generate all tasks you see
- There will always be atleast one task
- You always need a label field
- You always need at least one text field (text_1)
- If there are no text fields (for example images and text), output an empty array

Here are some examples of what you need to do.

Dataset and Structure:
{"sentence": "Value", "label": "ClassLabel", "idx": "Value"}
[
{'text_1': 'sentence', 'label': 'label', 'task_type' : 'Classification', 'index': 'idx'},
]
**Dataset and Structure:**
'{"premise": "Value", "hypothesis": "Value", "label": "ClassLabel", "idx": "Value"}'
[
    {
        "text_1": "premise",
        "text_2": "hypothesis",
        "label": "label",
        "task_type": "Classification",
        "index": "idx"
    }
]
Dataset and Structure:
{% endraw %}
</details>

This prompt above was used with [Qwen2.5-Coder-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ). Reasoning for this choice is that it was the best model
that I could run locally.
Hyperparameters used are below. Multiple calls are made and outputs are added together.
<details>
<summary> Prompting Code and Hyperparameters </summary>

def transform_call(key_value_mappings,prompt):
    # Define the prompt and other parameters
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    key_value_mapping_str = json.dumps(key_value_mappings)
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "developer",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt +f"\n{key_value_mapping_str}\n",
            }
        ],
        "response_format" : { "type": "json_object" },
        "n":3,
        "max_completion_tokens" : 1024,
        "seed" : 0,
        # https://www.reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/
        "temperature" : 0.2,
        "top_k" : 20,
        "top_p": 0.8,
        
    }
    
    # Make the API request
    # TODO: Add error handling for each type of error that could occur in this code
    # 500 codes, timeout, etc..
    response = requests.post(base_url, headers=headers, json=data)
    response_data = response.json()
    transform_json_strs = list(set([choice['message']['content'] for choice in response_data['choices']]))
    transforms = [task for transform_json_str in transform_json_strs for task in json.loads(transform_json_str)]
    # Applies transform
    extracted_tasks = set()
    for transform in transforms:
        try:
            # Dedupe step to remove identical tasks, doesn't remove mostly matched tasks..
            task_data = Task(data=transform, allowed_values=set(key_value_mappings.keys())).data
            task_json_str = json.dumps(dict(sorted(task_data.items(),key=lambda x:x[0])))
            extracted_tasks.add(task_json_str)
        except Exception as e:
            logging.warning(f"Failed to extract task {key_value_mappings.keys()} due to {e}")

    return [json.loads(task) for task in extracted_tasks]
</details>



<details>
<summary> Verification code </summary>

</details>

After it's all said and done, the result is a list of tasks which will eventually comprise the mega dataset.

## The Results

To better understand the impact of the use of LLMs in this context, two analyses were performed
1. performance
	1. recall for task extraction (i.e., what tasks were missed)
	2. recall and precision for task extraction correctness (i.e., how correct are tasks pulled out)
2. cost estimates (i.e., is this worth the effort)

Due to limited time, evaluation wasn't done over the entire dataset. Instead samples of 30 were chosen.
This seemed like a good balance between having meaningful statistics and tools versuses effort.

### Performance

#### Unlabelled datasets
<!-- explain dataset config before -->
Some of the time the prompt fails to find any tasks. Out of the 358 dataset & config pairs, 207 were parsed successfully
The parse rate is 57.82%. To better understand these failures 30 dataset & config pairs were sampled and assigned high level categories

<!--
<details>
<summary> Sample of Failed Tasks and Corresponding Labels </summary>
{%raw%}
task not clearly formatted: [all blimp tasks, https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k,HarmfulQA , https://huggingface.co/datasets/OpenSafetyLab/Salad-Data]
non-text (not an error): [Speech-MASSIVE, https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Tox21]
no label present: [cnn novel 125k,https://huggingface.co/datasets/edaschau/bitcoin_news  ]
error: [aegis,https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1,https://huggingface.co/datasets/argilla/medical-domain, https://huggingface.co/datasets/blinoff/kinopoisk,https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked, https://huggingface.co/datasets/harshi321/netflix-movies_shows,
https://huggingface.co/datasets/ai-forever/ru-reviews-classification, https://huggingface.co/datasets/rtweera/customer_care_emails]
{% endraw%}
</details>
-->

| Reason for Failure           | Number of Datasets |
|------------------------------|------------------|
| Unclear how to extract      | 17               |
| No text inputs present (not an error)      | 2                |
| No label present             | 2                |
| True Error                        | 9                |

Half of the "Unclear How to Extract" failures came from the [blimp dataset](https://huggingface.co/datasets/nyu-mll/blimp). Looking at the dataset without more context, I think failure makes sense. 
The task doesn't really have explicit labels, rather they are implicit. Additional information such as the readme would need to be provided.

The "True Error" class may be due to the lack of diversity in my prompt. In the errors a lot of the items were classification tasks, but that had lists as outputs. That might be a bit delicate to handle since those outputs will need additional post processing.
The tasks that failed mostly seemed to be multi-label. Including additional metadata like a sample of the data or the readme might help in these cases.

#### Correct Labels

The 207 datasets that did parse lead to 310 tasks. 30 of these were randomly sampled for correctness.

<!--
Notes on the 30 samples of correct labels
Correct, 1 of n tasks, ('LLM-jp-Toxicity-Dataset', 'default', {'index': 'id', 'label': 'obscene', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of n tasks, ('informes_discriminacion_gitana', 'default', {'label': 'tipo_discriminacion', 'task_type': 'Classification', 'text_1': 'sintetico', 'text_2': 'text', 'text_3': 'intervencion'})
Correct, 1 of 1, ('APEACH', 'default', {'label': 'class', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of n tasks, ('Russian_Sensitive_Topics', 'default', {'label': 'drugs', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of n, ('Mawqif_Stance-Detection', 'default', {'index': 'ID', 'label': 'stance', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of 1, ('stsb_multi_mt', 'fr', {'label': 'similarity_score', 'task_type': 'Regression', 'text_1': 'sentence1', 'text_2': 'sentence2'})
Correct, 1 of 1, ('myanmar_news', 'default', {'label': 'category', 'task_type': 'Classification', 'text_1': 'text'})
Incorrect, misunderstood task, ('covid19_emergency_event', 'default', {'label': 'all_events', 'task_type': 'Sequence Labeling', 'text_1': 'text'})
Correct, 1 of 1, ('glue', 'stsb', {'index': 'idx', 'label': 'label', 'task_type': 'Classification', 'text_1': 'sentence1', 'text_2': 'sentence2'})
Incorrect, mistook multilabel classification for classification, ('goodreads-bookgenres', 'zeroshot-labels', {'label': 'predicted_genres', 'task_type': 'Sequence Labeling', 'text_1': 'Book', 'text_2': 'Description'})
Correct, 1 of 1, ('Salad-Data', 'base_set', {'index': 'qid', 'label': '1-category', 'task_type': 'Classification', 'text_1': 'question'})
Correct, 1 of 1, ('glue', 'cola', {'index': 'idx', 'label': 'label', 'task_type': 'Classification', 'text_1': 'sentence'})
Correct, 1 of 1, ('babi_nli', 'indefinite-knowledge', {'index': 'idx', 'label': 'label', 'task_type': 'Classification', 'text_1': 'premise', 'text_2': 'hypothesis'})
Incorrect, mistook multilabel classification for classification, ('fineweb-c', 'gsw_Latn', {'index': 'id', 'label': 'educational_value_labels', 'task_type': 'Sequence Labeling', 'text_1': 'text'})
Correct, 1 of n, ('MentalManip', 'mentalmanip_con', {'index': 'id', 'label': 'vulnerability', 'task_type': 'Classification', 'text_1': 'dialogue'})
Incorrect, mistook multilabel classification for classification, ('fineweb-c', 'gmh_Latn', {'index': 'id', 'label': 'educational_value_labels', 'task_type': 'Sequence Labeling', 'text_1': 'text'})
Correct, 1 of 1,('patent-classification', 'abstract', {'label': 'label', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of 1 ,('MainframeBench', 'multiple_choice_question', {'index': 'Unnamed: 0', 'label': 'answer', 'task_type': 'Classification', 'text_1': 'prompt', 'text_2': 'question'})
(Partially) Correct 1 of 1, arguably had extra text fields,('french_book_reviews', 'default', {'label': 'rating', 'task_type': 'Regression', 'text_1': 'book_title', 'text_2': 'author', 'text_3': 'reader_review'})
Correct, 1 of 1 ,('afrihate', 'kin', {'index': 'id', 'label': 'label', 'task_type': 'Classification', 'text_1': 'tweet'})
Correct, 1 of 1 ,('afrihate', 'zul', {'index': 'id', 'label': 'label', 'task_type': 'Classification', 'text_1': 'tweet'})
(Partially) Correct, 1 of 1, ('TurkishHateMap', 'cities', {'label': 'label', 'task_type': 'Classification', 'text_1': 'baslik', 'text_2': 'text'})
Correct, 1 of 1,('alloprof', 'queries', {'index': '__index_level_0__', 'label': 'answer', 'task_type': 'Sequence Labeling', 'text_1': 'text'})
informes_discriminacion_gitana ('Russian_Sensitive_Topics', 'default', {'label': 'offline_crime', 'task_type': 'Classification', 'text_1': 'text'})
Incorrect, mistook multilabel classification for classification,('goodreads-bookgenres', 'original-genres', {'label': 'Genres', 'task_type': 'Sequence Labeling', 'text_1': 'Book', 'text_2': 'Description'})
Correct, 1 of 2, ('multiclass-sentiment-analysis-dataset', 'default', {'index': 'id', 'label': 'label', 'task_type': 'Classification', 'text_1': 'text'})
Incorrect, mistook multilabel classification for classification, ('fineweb-c', 'fra_Latn', {'index': 'id', 'label': 'problematic_content_label_present', 'task_type': 'Classification', 'text_1': 'text'})
Correct, 1 of 1, ('babi_nli', 'three-supporting-facts', {'index': 'idx', 'label': 'label', 'task_type': 'Classification', 'text_1': 'premise', 'text_2': 'hypothesis'})
Correct, 1 of 1, ('babi_nli', 'conjunction', {'index': 'idx', 'label': 'label', 'task_type': 'Classification', 'text_1': 'premise', 'text_2': 'hypothesis'})
Correct, 1 of N,('toxic-chat', 'toxicchat1123', {'index': 'conv_id', 'label': 'jailbreaking', 'task_type': 'Classification', 'text_1': 'user_input', 'text_2': 'model_output'})
-->

6 of 30 failed, the mistake for all of them was use of the incorrect task. The model confused sequence labelling for classificaiton. All of these tasks had multiple
labels, which may part of the reason for this failure.
There were also 2 questionable inclusions of other text fields that weren't necessary to the task,
This gives us a precision of 80% for the task_labelling, 93.33% precision for correct task extraction.

Overall performance is decent from what I've seen. A larger sample or ideally evaluating the entire population would be required to confirm this.

### (Human) Time Efficiency

In terms of efficiency there is no clear answer. The following numbers are all estimates.

To draft mappings it took me on average 1.5 minutes, however this was with a very inefficient process that involved
manually typing out a JSON. Likely with a decent UI this process could take 30-40 seconds (educated guess).
For 215 datasets this would take about 5 hours and 20 minutes. Not only is this a long time, but personally I find it
very boring.

Subjectively creating code, a prompt, and rules to verify that prompt is much less discouraging than sifting through hundreds of datasets manually.
That being said I'm not convinced it's necessarily more efficient, I estimate it took about three hours to get the pipeline working properly
(coding, pilots) and an hour to verify. The automated process is likely more scalable too.

### Cost Efficiency

Beyond the time required, cost is another consideration. Here are rough estimates:

The prompt above took about 500 watt per hour and each run took 1h30 min. For me that translates to 14 cents per run.
In total I did 8 runs, at a cost of $1.02 . That however doesn't account for the time to create the tool, verify the output.
Assuming a developer costs [$60 an hour](https://www.indeed.com/career/python-developer/salaries) and an annotator to verify
costs $12 the total cost would be $193.02


If you can pay an annotator and reviewer $12, assuming equal time required for both the cost would be ~$128. Not only is this 
cost lower, but the performance of a human annotator and reviewer is significantly higher. While model time is not expensive, it isn't clear how much
would be required (if possible at all) to get human like performance.

## Potential Improvements

Post analysis, it seems possible simple improvements could improve performance and make LLMs
more viable for this use case. Specifically by:
1. using of a voting mechanism rather than raw aggregation
2. including more information in the prompt
3. adding a multilabel task
4. increasing the number of examples from 1 to 5-9 (guess, would need to hp tune)

## In Conclusion

LLMs for ETL seem viable, in scoped down & verifiable use cases. While 
this may seem like a more efficient option, the cost
of designing rules for the system needs to outweigh the cost of creating rules and 
manual verification after the fact. For this use case, the ability to scale is interesting
and worth the comparably worse performance compared to human annotators.
For a team with dedicated annotators or where data loss and issues can't be tolerated this approach may not be worth it. However if you are alone
using LLMs may be preferable to manually creating hundreds of mappings.