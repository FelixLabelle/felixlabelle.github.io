# Practical Tidbits: Curating a List of Sensible Words

Recently I worked on a, shortly to be published, project about finding patterns in randomly generated strings ([YubiKey](https://en.wikipedia.org/wiki/YubiKey) strings).
That project required a vocab (i.e., list of words) and I was not impressed with the ones I found online. 
This isn't the first time I've been frustrated by poorly selected vocabs;
word games can suffer from a similar problem. I find it frustrating when I'm playing a game 
and the solution is an archaic or arcane word. You might say that's part of the game,
but some words from the engineering domain or other archaic words won't work so it makes for a jagged experience.

For these reasons I think a curated word list might be useful beyond my use case. The end goal of this work was to have a labeled list of words 
you could pick from à la carte, or at least a go/no-go that excludes "weird" words.

## Problem

For my analysis on YubiKey strings I just wanted a list of contemporary words with a large enough English-only vocab 
which aligns with what most people would know.
The YubiKey output itself doesn't have words so to speak (unlike most text corpora), so an external reference is needed. When it comes to vocabs, there are some common
resources, but I haven't found any I was satisfied with:

* [Scrabble](https://gist.github.com/deostroll/7693b6f3d48b44a89ee5f57bf750bd32) has so many nonsense/bullshit words. Acronyms or just weirdness. It's mind-boggling; AA, ZA, BUMFUZZLE are all valid words. I want to say more than 30% of the vocab is nonsense in my estimate
* [WordNet](https://wordnet.princeton.edu/) (or [actively maintained English open source alternatives](https://github.com/globalwordnet/english-wordnet)) are similarly plagued with acronyms and other entries that are less interesting. They do have usage labelled.
* [Oxford 3000](https://github.com/jnoodle/English-Vocabulary-Word-List/blob/master/Oxford%203000.txt)/[5000](https://github.com/jnoodle/English-Vocabulary-Word-List/blob/master/Oxford%205000.txt) are good resources, but limited in size. Some common words like "fuel" are missing. To match [most people's vocabs 20k+ words are needed (likely in excess of 30k+ since people typically know that many lemmas, let alone words)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4965448/) 
* Sampled corpora (e.g., [wordfreq](https://github.com/rspeer/wordfreq)) are good and have additional useful information like use frequency, but they are filled with noisy examples, antiquated words, or "improper" words (slang, etc.).
* [12Dicts](https://wordlist.aspell.net/12dicts/) is the closest thing to a large vetted resource I could find, but its selection criteria differ from mine. I noticed there were a large number of acronyms, abbreviations, or multi-word entries. This could be solved, but would require some additional logic and I want to avoid heuristics if possible.

The alternative is to curate our own vocab.

## Solution

The end goal is a 30k+ word vocab with only "sensible" words. We'll get more into what that means shortly.
I chose to start with [wordfreq](https://github.com/rspeer/wordfreq) and use an LLM + prompt to filter "weird" words. This approach provides:
1. a larger set of words
2. well-delimited, individual words
3. ability to select which words to drop 
4. additional metadata around word usage

This was a bit of a detour from the YubiKey analysis, so I chose to prompt an LLM. It's easy, works relatively well, and a giant 
word machine trained over a fair share of all existing human text is likely to have enough context to 
know about the senses and common usages of words.
The prompt defines categories of words to exclude, keeping only words that do not fall into those categories. For any words that are difficult
to classify with a given prompt, there is a catch-all "review" label. The classes are the following:
<br>
<details markdown="1">
<summary>Filter Classes</summary>
<pre class="code">
'obscure_borrowing', 'misspelling', 'nonstandard_spelling',
 'artificial_derivation', 'review', 'keep', 'abbreviation',
 'foreign_only', 'obscure_historical', 'archaic', 'obscure_dialect', 
 'nonsense', 'proper_noun', 'obsolete'
</pre>
</details>
<br>

 
The prompt outputs a word -> class mapping over every word provided to it.
Guardrails are used to verify every word is present and the labels returned are valid.
I tried other formats (e.g., lists of dropped words), but this one worked the 
most consistently. Other output formats didn't pass their guardrails as often and would require multiple 
attempts to work.
The words to classify are provided as a list. I used fixed-size batches of 20 words
to speed things up. I didn't ablate the size and don't know the effects 
of word order or list size on performance.

<br>
<details markdown="1">
<summary>Filter Prompt</summary>
<pre class="code">
# English Vocabulary Quality Filter

You are a lexicographer reviewing candidates from an English word-frequency list.

Your task is to produce a high-quality vocabulary of **established, contemporary English words** suitable for general-purpose language processing.

The input contains lowercase word candidates, not a validated dictionary. It includes names, foreign text, abbreviations, and errors. Lowercasing or high frequency does not make an entry an ordinary English word.

**Prioritize precision over completeness**, but do not mistake uncommon vocabulary for invalid vocabulary.

## Classification labels

Assign every word exactly one label:

- **KEEP:** An established, correctly spelled, in-scope contemporary English word, including legitimate specialist terms and the name-derived senses explicitly allowed below.
- **DROP:** An entry excluded by any rule below, including names and named-group affiliations, even if familiar or correctly spelled.
- **REVIEW:** A potentially in-scope word with a specific unresolved question about its spelling, meaning, or contemporary usage. This is an exceptional fallback, not a category for recognizable exclusions or ordinary words with multiple meanings.

Classify lexical meanings, not every entity a spelling could refer to. A word is not excluded merely because it can also name a person, team, or place. Apply the common-word distinction in rule 7 before rejecting an entry as a name.

The policy decisions and examples below are binding, not unresolved questions. When categories overlap, an explicit KEEP or DROP example takes precedence over a broad rule. Do not REVIEW a case whose eligibility has already been settled here.
For each input, check the explicit DROP examples first, then the explicit KEEP examples, then apply the general rules. An excluded borrowing is not rescued by calling its meaning technical: `chil` is DROP, not an alternative spelling of the ordinary word `chill` (KEEP).

## Rules

### 1. Standard English validity

KEEP words that have established meanings and recognizable grammatical uses in contemporary English.

DROP:
- Nonsense letter combinations.
- Misspellings and typographical artifacts.
- OCR and transcription errors.
- Arbitrary sequences of letters.
- Words whose only justification is inclusion in another spell-checking list.
- Fragments that only occur inside a foreign phrase or as a prefix/suffix, not as independent English words.
- Spellings that require restoring missing punctuation, such as `dont` or `whats`.
- Deliberate nonstandard or humorous respellings, even when dictionary-attested: `laff`, `fotos` → DROP (`nonstandard_spelling`). Slang attestation does not make a nonstandard spelling acceptable.

Classify the exact input spelling, not an implicitly corrected word. For example, `superheros` is DROP (`misspelling`); the standard plural is `superheroes`. `nauseum` is also DROP, not a valid word merely because it resembles the phrase `ad nauseam`.

A word being unfamiliar is not sufficient reason to drop it.

### 2. Historical and archaic language

DROP words used almost exclusively in historical, archaic, or obsolete English.
This includes archaic pronouns and verb inflections such as `thee`, `hast`, `dost`, `knoweth`, `runneth`, and `mayest`; a valid historical conjugation is not a contemporary inflection.

Examples:
- `forsooth` → DROP
- `prithee` → DROP
- `quoth` → DROP

KEEP old words that remain in ordinary contemporary usage.

Do not drop words merely because they have historical origins.

However, explicitly DROP rare historical, antiquarian, or craft-object senses with insufficient broader contemporary currency, even when a dictionary lists them or modern historical writing mentions them. Do not rescue an otherwise excluded name through such a sense.

`mita`, `frater`, `logan`, `spencer`, `firestone`, `monteith`, `bibliotheca`, `bethel`, `telfer`, `samarra`, `bourg` → DROP (`obscure_historical`). This is not a ban on ordinary modern words for old things, such as `chariots` or `muskets`, or on established contemporary occupations and tools.

### 3. Morphology and inflections

KEEP established forms of valid words, including:
- Regular and irregular plurals.
- Verb conjugations.
- Comparative and superlative adjectives.
- Established adverbs and derived adjectives.

Apply all exclusion rules to the root and its forms: a plural or derivative of an excluded name or named group is still DROP. Conversely, ordinary inflections such as `scientists`, `alerts`, and `aborts` are KEEP, not REVIEW.

DROP mechanically generated or highly unnatural derivatives with no credible independent usage.

For example:
- `abandoned` → KEEP
- `abandonedly` → DROP

The fact that an affix can grammatically attach to a root does not establish the resulting word as legitimate.

### 4. Rare but legitimate words

KEEP uncommon words if they have established contemporary usage and satisfy the scope rules. Narrow technical/professional use is allowed by rule 5; rare historical/craft senses, narrow dialect/slang, obscure borrowings, nonstandard respellings, and spoken clippings remain excluded by their respective rules.

Examples:
- `obdurate` → KEEP
- `abecedarian` → KEEP
- `cerulean` → KEEP

Do not use word frequency alone as a measure of validity.
Accept standard regional spellings, including British and American variants: `industrialisation`, `visualised`, and `misdemeanours` are KEEP.

### 5. Technical terminology

KEEP established contemporary scientific, mathematical, computing, medical, musical, philosophical, and other technical/professional terms, INCLUDING highly specialized ones. No general-reader familiarity or textbook-level frequency cutoff applies. A word used mainly by specialists is not for that reason DROP or REVIEW.

This includes established technical adjectives, inflections, notation words, and clinical ellipses used as lexical terms. Explicitly KEEP `cohomology`, `eidos`, `loral`, `lutea`, `previa`, `naturale`, `casse`, `memoria`, and `tremblant`. In particular, `previa` and `lutea` are allowed technical terms, not rejected spoken clippings; `naturale` is allowed musical terminology, not an excluded obscure cultural borrowing.

Still DROP obsolete scientific nomenclature, historical measurement units no longer in meaningful use, scientific symbols, and mere Latin taxonomic names that have not become English common words. Technical status does not override the explicit rare-borrowing DROP examples in rule 6 or rescue other nontechnical policy exclusions.

Named technologies without an independent lexical sense remain DROP: `postgres`, `javascript`, and `bluetooth` are names, not generic technical vocabulary. Established name-derived verbs such as `photoshop` qualify under rule 7.

Examples:
- `algorithm` → KEEP
- `mitochondria` → KEEP
- `zygote` → KEEP
- `zirconium` → KEEP
- `abhenry` → DROP
- `zygapophysis` → KEEP
- `phosphatidylinositol` → KEEP

### 6. Regional words, slang, and borrowings

KEEP ordinary contemporary informal words and widely naturalized English words of foreign origin. Standard British/American spelling variants are allowed; foreign origin or profanity alone is not disqualifying.

Explicitly DROP narrow dialect/Scots forms, city-specific or subcultural slang, rare regional/literary survivals, weakly established derivatives, and nonordinary inflections. Attestation in a regional dictionary, isolated quotation, lyric, or online subculture does not override this scope exclusion.

`richt`, `gaither`, `hele`, `sele`, `roose`, `kelty`, `levant`, `dess`, `homi`, `linky`, `goldy`, `youngs` → DROP (`obscure_dialect`, `artificial_derivation`, or `nonstandard_spelling`, as appropriate).

Also explicitly DROP less-common cultural, culinary, musical-instrument, folklore, craft, and regional natural-history borrowings that have not become broadly naturalized English vocabulary. Being mentioned in English specialist writing, on a foreign menu, or in a dictionary is not sufficient. These words are excluded even if a plausible English borrowing sense exists:

`palu`, `lotte`, `kaen`, `xiao`, `kodama`, `homa`, `salle`, `shino`, `mazur`, `chil`, `smee`, `pono` → DROP (`obscure_borrowing`).

Do not extend this exclusion to all foreign-origin words or to the established professional terminology allowed in rule 5. For example, `piano`, `pasta`, and `coffee` are naturalized English words; `naturale` is an expressly allowed technical term, whereas `kaen` is an expressly excluded borrowing.

### 7. Names, named groups, and abbreviations

Apply these decisions in order, honoring explicit policy examples:

1. For this vocabulary, explicitly DROP `letterman`, `democrat`, `democrats`, `republican`, and `republicans` using `proper_noun`; occupational or generic political senses do not override these policy exclusions.

2. KEEP established name-derived method, style, era, scientific, and philosophical adjectives for their non-membership senses: `socratic`, `pythagorean`, `italianate`, `edwardian`, `kantian`. This also includes scientific-method/theory adjectives such as `darwinian`. Conventional capitalization or derivation from a person, nation, or named school does not exclude these accepted senses. This does not automatically admit member-only plurals such as `kantians` or ordinary nationality/language labels such as `italian`.

Also explicitly KEEP the following previously ambiguous name-related lexical senses:
- `leonid`: an individual meteor of the Leonids shower.
- `gentiles`: people outside a religious community, especially non-Jews.
- `tarzan`: a strong, agile person; the established allusive common-type sense is allowed.
- `lokpal`: an anti-corruption ombudsman or public-office/body term.
- `rolf`: perform the established deep-tissue treatment, despite its service-mark origin.
- `vidalia`: the sweet-onion food type, despite its certification-mark status.

These allowances do not admit arbitrary named people, ethnic groups, religions, organizations, or products. They admit an established lexical sense, not the name alone. A trademark, certification mark, named institutional origin, or character allusion is not a veto when such a sense qualifies. Generic product-type nouns and established brand-derived verbs such as `photoshop`, `google`, and `xerox` can therefore be KEEP; a mere brand/company name such as `microsoft` is still DROP.

3. Otherwise, KEEP genuine common-word homographs: an established, independently useful contemporary common-noun, verb, adjective, or adverb sense takes precedence over any name sense, provided that sense is not excluded by rules 2, 6, or the clipping/spelling rules. It need not be the most frequent or most famous use. Do not discard an ordinary plural just because a sports team uses it, or an occupation just because it is also a surname.

KEEP examples:
- `harry`, `jersey`, `robin`, `smith`, `warren`, `rowan`, `python`, `sears`, `snickers`.
- `madden` (infuriate), `mercury` (element), `java` (coffee).
- `atlas` (collection of maps, not only the mythological name), `utopia` (ideal society), `proctor` (examination supervisor).
- `king`, `dean`, `cooper`, `mason`, `carter` (haulier), `sawyer` (woodcutter), `wheeler` (a wheeled vehicle or a wheelwright), `august`, `senate`, `parliament`.
- `china` (porcelain/tableware), `turkey` (bird), `polish` (make smooth or shiny).
- `orioles` (birds), `cardinals` (birds or clergy), `giants` (huge beings or exceptionally large things), regardless of team-name uses.

Generic categories of people are not named affiliations: `scientists`, `teachers`, `worshippers`, and `atheists` are KEEP. The exclusion concerns membership in a specifically named group, not all words describing people or beliefs.
Likewise, generic political concepts such as `democracy`, `democracies`, `republic`, and `republics` are KEEP. Sharing a root with an excluded party label does not make a distinct common word a named affiliation.

KEEP lexicalized common words originating from names, such as `cardigan`, `sandwich`, and `diesel`. Generic foods, drinks, and materials named after places are also KEEP when established as independent common nouns: `amontillado` (sherry), `gouda` (cheese). Spelled-out contemporary unit names such as `newton` and `watts` are KEEP, not measurement symbols.

4. If there is no qualifying lexical sense or express allowance above, DROP the following using `proper_noun`, not REVIEW:
- Personal names, surnames, nicknames, and named fictional or mythological characters.
- Geographical names, brands, companies, products, software names, and titles of specific works.
- Nationalities, ethnic groups, demonyms, languages, and their corresponding adjectives: `russian`, `japanese`, `british`, `arab`, `rwandans`, `occitan`, `hokkien`.
- Named religions, denominations, political parties, organizations, teams, and fandoms, including their members, followers, affiliation adjectives, and derived forms: `buddhism`, `mormon`, `mormons`, `tories`, `scientologists`, `carmelites`, `beliebers`.

These exclusions apply even when an entry is conventionally lowercase, has a dictionary definition, or is a plural. Do not rescue a name through an obsolete, invented, or excluded rare historical/craft sense. Familiarity alone does not establish a qualifying lexical sense; conversely, unfamiliarity does not exclude an established specialist term.

DROP initialisms, abbreviations, and scientific or measurement symbols using `abbreviation`, including their plurals: `prof`, `bdsm`, `rpgs`, `suvs`, `ieds`, `idps`.

Explicitly prohibit spoken clipped/shortened forms as well, even when pronounced as words, dictionary-attested, pluralized, or common in a profession: `perc`, `carbo`, `biomed`, `recs`, `juco` → DROP (`abbreviation`). Spoken usage and lexicalized-clipping claims do not override this policy. Fully naturalized acronym-origin words such as `radar` and `laser` are not these clipped forms and remain KEEP; the explicitly allowed clinical terms `previa` and `lutea` also remain KEEP.

### 8. Uncertainty

Do not fabricate definitions or usage examples.

First apply the explicit policy decisions, inclusion rules, and exclusions. KEEP qualifying contemporary lexical senses, including the allowed name-derived and specialist senses. DROP entries whose only senses are excluded names/affiliations, rare historical/craft senses, narrow dialect/slang, obscure borrowings, nonstandard spellings, clippings/abbreviations, foreign-only forms, or archaic language.

Unfamiliarity, multiple senses, standard regional spelling, regular inflection, and specialist usage are not by themselves reasons for REVIEW. Do not require external verification of familiar vocabulary. Never use REVIEW merely to reopen an expressly allowed or prohibited case.

Use REVIEW only when a specific lexical question genuinely remains unresolved after applying the rules, not merely because a word may be a name, is uncommon, or needs verification. Do not invent uncertainty to populate this category: returning no `review` labels is normal. Do not output explanations.

## Output format

Return one flat JSON object mapping every input word to exactly one lowercase class string:

- `keep` for accepted words.
- A rejection code below for rejected words, never the generic label `drop`.
- `review` for genuinely unresolved words.

Only these rejection codes are permitted:

`archaic`, `obsolete`, `misspelling`, `nonstandard_spelling`, `nonsense`, `artificial_derivation`, `proper_noun`, `abbreviation`, `foreign_only`, `obscure_dialect`, `obscure_historical`, `obscure_borrowing`.

Example: {{"atlas": "keep", "letterman": "proper_noun"}}. Do not use wrapper fields, arrays, nested objects, or explanations.

**Output requirements:**
- Classify every input word exactly once. The object keys must match the entire input with no omissions, duplicate keys, or extra words.
- Emit keys in input order and verify their count matches the input length. An unresolved decision must be `review`, never an omitted entry.
- Preserve the original spelling.
- Never introduce words not present in the input.
- Never silently skip entries.
- Before returning, check KEEP for policy exclusions, DROP for allowed common-word/name-derived/specialist senses, and REVIEW for decisions already resolved by these rules.
- Return JSON only, without Markdown or additional commentary.

## Words to classify

{}
</pre>
</details>
<br>

The prompt itself is based on a rough list of criteria I drafted. The initial prompt was then generated and 
iterated upon by an agent (OpenCode, GPT 6.1 Sol Medium Effort). I set up a loop where GPT 6 Luna (Codex, Low) classified words from the entire corpus,
then a sample of 1000 words was pulled. Any word the model previously flagged as being misidentified 
would be pulled in too. Then a larger model (6.1 Sol High, OpenCode) analyzed the latest outputs 
of the sample and previously failed words and tweaked the prompt to "address those".
This process was repeated 3 times. After each iteration I looked at the work and outputs of this 
tuning process. I chose to stop after 
3 iterations as the prompt had hit a point of diminishing returns IMO and was just 
shuffling labels around without making real improvements to keep vs drop.

Two models were used with that tuned prompt: GPT 6 Luna and GPT 6.1 Sol, both through Codex.
I committed the cardinal sin of not using a "decision model" because I don't have personal access to Jev 
and my homelab doesn't have any free compute at the moment to run [one of the hundreds of replication attempts](https://hanxiao.io/all-about-jev/#v=table&cat=model).
Moreover I'm still investigating how well decision models generalize to new data and tasks.

Why Codex?
Annotation tasks like this are an easy way of making sure my weekly quota doesn't go to waste.

I chose to use "low" reasoning for both models, given the trade-off between 
cost and processing time wasn't worth it.
One note, is that myself and [others have observed reasoning doesn't make a large or noticeable difference 
in performance for **certain tasks** across model families](https://x.com/skalskip92/status/2094884897416479162).
[Author above also has a subjective measure for this](https://roboflow.com/blog/gpt-6-astra-vision?utm_source=chatgpt.com).
The example provided is for image tasks, but I've noticed the same in text-only classification like 
reranking.


After the models are run over wordfreq, vocabs are saved as CSV files, with word, class, and [Zipf score](https://en.wikipedia.org/wiki/Zipf%27s_law). There is a fused vocab as well that only contains
the intersection of the models' keep labels. [Code and vocab CSVs can be found here](https://github.com/FelixLabelle/quality_vocab_project).

## Analysis (i.e., what does it look like?)

Without ground truth labels, the analyses that can be run are limited. I started with the overall
distributions, compared the models' outputs, and then sampled words to get a subjective sense of quality.
Zipf score is a logarithmic score where higher scores map to higher frequency. Basically bigger number means more common, smaller number less common. Below is a formula
that describes it:

$$
\text{Occurrences per million words} = 10^{z - 3}
$$



### Data Summaries

Both models kept about 82k words from the shared candidates. That's higher than the vocab size I was
aiming for. The plots below show how the kept and rejected words
are distributed by wordfreq's Zipf score. 

![Luna at low reasoning: overlapping frequency histograms for kept and rejected words](/images/luna_low_plot.png)
![Sol at low reasoning: overlapping frequency histograms for kept and rejected words](/images/sol_low_plot.png)

The distributions look quite similar. Both models are more likely to reject low-frequency words,
while kept words make up more of the higher-frequency bins. That said rare words can be kept 
a fair amount of the time shows frequency alone isn't enough to decide
whether or not a word is useful (assuming the model is correct).

### Inter-model Comparisons

![Frequency distributions for kept and rejected words in the joint vocabulary](/images/luna_+_sol_coarse_intersection_plot.png)

Requiring both models to keep a word brings the joint vocab to about 79k words without changing
the overall shape much. However that still doesn't tell us whether
the labels are right.

Similar distributions don't necessarily mean the models are keeping the same words.

![Three-class comparison of the two models, showing keep, review, and drop counts](/images/coarse_correlation_matrix.png)

The Sol and Luna runs have 175,055 shared words.
The coarse matrix compares how Sol and Luna align on keep, drop, and review, with Luna on the rows and Sol on the columns.
For coarse labels, the models agree on 168,746 words, or 96.40% of the time.
Cohen's kappa is 0.928 for coarse alignment.
That's an encouraging level of consistency, especially given the number of words.

![Fine-grained model comparison across all fourteen classification labels](/images/finegrained_correlation_matrix.png)

The fine-grained matrix shows where the individual labels differ, with Luna on the rows and Sol on the columns.
Cohen's kappa is 0.842 for fine-grained labels, which is quite a bit higher than I thought. It doesn't mean much,
but it made me smile.
Most disagreements concern which rejection or review label to use, so not much more signal to be gleaned here.

### Subjective Quality

Agreement is useful, but both models could be making the same mistake.
Realistically it's hard to label a vocab of this size; as a lone annotator
it would almost certainly contain idiosyncrasies. That being said,
a ground truth is needed to understand where issues might lie.


I chose to annotate a sample. Rather than randomly sample, I sampled words based on the Zipf score, cutting the vocab into 8 bands and taking up to 150 samples from each band.
Bands with less than 150 samples (i.e., the most common words) will have all their words sampled.
The idea is that this approach makes it possible to see if there is a Zipf score
where model quality starts to break down.

In total 794 words were sampled and annotated.
I will say that labeling the long tail is painstaking, so I understand the models making mistakes.
My process involved several dictionaries and example usage sites for any word I was unfamiliar with.
The labeled data can be found in this [CSV](https://github.com/FelixLabelle/quality_vocab_project/blob/main/analysis_outputs/labelled_150_7_band_sample.csv).


When it comes to the quality 
of the vocab, I'm primarily interested in 
the rate of false positives (words that should have been dropped, but were kept).

| Zipf Band | Human-Labelled “Keep” | Human Rejections Among Words Both Models Kept |
|---|---:|---:|
| 3–4 | 120/150 (80%) | 1/119 (about 1%) |
| 2–3 | 63/150 (42%) | 2/65 (about 3%) |
| 2 and below | 43/150 (29%) | 4/44 (about 9%) |

Above is a table with the false positive rate per band for the intersection of the models. 
It makes it easier to see/understand how quality degrades.
Based on the sample I reviewed, 
I would assume any word with a Zipf score of 3 or higher is relatively likely to be well labeled. 
This maps to about 20k words kept.
A score of 2.5 is about 35k. This still seems like a good cutoff. 2 is where quality starts 
appearing degraded to the point of being a questionable choice. I would also say don't 
trust the fine-grained labels for rarer words. I disagreed with a number of those as well,
for both models.

## Caveats

The resulting vocab was meant to be used in a downstream project and this effort is not a project in its own right.
I didn't do a deep dive into classifier performance beyond 
sampling the high-level categories (keep, drop, review) and some limited iterating on the prompt.
There may be jagged edges, but based on my sampling (see analysis above) I'm relatively
confident the vocab will be workable.

As a consequence of the classification scheme and language itself having some ambiguity, words may fall into multiple categories.
I cared about just keep vs drop, so the scheme may be what's causing the fine-grained labels to be unreliable. Moreover,
the models only output one label per word and it's unclear if the output is repeatable (a different run could give different results).
Couple that with batches of size 20, repeatability is questionable at best.

There is also a blend of British and American English and no way of separating the two. Not a bad thing,
just worth noting if that matters to your use case.

WordFreq froze in time pre-LLM (2021) to avoid biasing by LLMs. While this is fine for my purposes,
there are caveats especially if you are using this resource in the future (post 2026).
Not only does language change over time, but it [appears human word usage has even changed since LLMs have been introduced](https://arxiv.org/html/2409.01754v1).
I still think this resource is relevant, but you may want to run this procedure over a more up-to-date text corpus if word frequency matters to you.
If human word usage converges towards that of LLMs (or vice-versa), pulling a web corpus might not be a bad idea.

## Conclusion and Next Steps

 I think the joint vocab when keeping the top 35k most frequent "keep" words is fit for purpose. It's large enough and looks good subjectively.
Hopefully others find it useful as well!

That said I do think an even larger vocab could be interesting for downstream analyses or games. 
However the filter isn't quite good enough yet when it comes to rarer words. I think there are a couple of routes for improvements:
* Additional signal could help, e.g., WordNet descriptions, word frequency, 12Dicts information about sources, or contexts in which the word is found
* Changing the prompt format to be on individual words, not batches
* Inclusion of confidence or in general uncertainty
* Improving the labeling scheme, words can be part of multiple classes. I also think a catch-all like keep might be leading to confusion. Multi-label or even multi-class (e.g., archaic vs contemporary, domain, English vs British vs common) might make more sense for this task, TBD
* Better quality data for evaluation, either existing datasets if using the multiple class approach or a more rigorous methodology if such data doesn't exist
* Further exploring fine-tuning/specialization

 This isn't an urgent need on my end, so will likely be on the back burner until I need a break from my longer running experiments or I need a larger vocab for another project.
