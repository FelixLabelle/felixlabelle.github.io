# A Dream: An Easy Way to Work with Documents and (implicitly) Structured Text

Writing can be hard, so here's a light post about working with text in documents while I finish researching, testing, and writing about textual similarity methods.

## Context

In NLP there are a lot of applications that use text contained within documents. 
For example, since LLMs as an API have become accessible, more projects that use RAG have come about.

When working with documents, there are two things I think that could be better
1. a better way of extracting text that isn't custom for every task
2. a common format for downstream applications that preserves inherent structure in documents

### Issue 1: The State of Text Extraction

Text extraction is currently adhoc for each new application and the documents within it.
For example, an enterprise chatbot might need access to documents across several departments. The documents queried might be word documents, pdfs, web pages, emails, or 
a combination of these. 

This might sound all good and dandy, but to be able to search each of these documents you will need to write a text extraction (and chunking) pipeline.
Even within a specific format, say PDFS, you might have different types of documents (e.g., TPS reports and API Documentation).

For some formats like PDFs just getting text out can be difficult. It is hard to differentiate between different types of text like the caption of an image,
verses plain text. Tables could be pulled out as text and mangle the main text without proper  care. Text could even be in an image an be inaccessible to a user (if not using the correct tools).

This means that for each project, we need to build individualized text extraction code and post-processors. This is time consuming and any mistakes here could (and probably will)
effect downstream performance.

### Issue 2: Inherent Structure in Text

Issue 1 is plain as day if you have ever done any type of data munging. The more subtle issue is that regardless of how well text extraction is done,
we are inherently removing structure from text. All documents have a structure that gives us as end consumers information. Think of a book, it has 
chapters. Just knowing what chapter a paragraph is in gives you additional context that could be invaluable for a downstream task like retrieval.

We could add relevant metadata in an adhoc manner, but this would require implementing this logic for each document type and there are no guarantees
a model would be able to easily process it. This text is likely OOD.

## Why Yeeting Documents into Models and Hoping they Understand Isn't a Solution

As multimodal models and models with long (100K+) contexts have become commercially available, there has been a tendency
to just throw more and more information into those long contexts. I assume some readers have put 2 and 2 together and wondered
why we couldn't just add whole documents (whether as images or text) into prompts and hope that works.

The answer is that 2 + 2 = 5. I think use of models in such a blind way is a bad idea for a number of reasons:

### We Don't Know How Longer Contexts are Used

WIP

### Brittleness of Prompts

WIP

### Degradation of Performance in Shorter Context

WIP

### Do Commercial Models Support Multiple Images

WIP

### Cost

WIP

### Not Reusable

WIP

## The Idea

I think one solution to both these problems is a universal data format (UDF).

This file format would have tooling to produce a share format that includes 
1. Context
	1. Type of document (report, book, web page, etc..)
	2. Intent of the document
	3. Summary
2. Parse tree/graph
	1. Textual structure
		1. chapters
		2. headers
		3. lists
		4. parapraphs
		5. sentences
		6. references
		7. other types of structure (if I missed any)
	2. Different non-text elements
		1. Images
		2. Tables
		3. Figures
		4. External references
	3. Interactions between text and non-text elements
		1. References
		2. Complimentary Information
3. Relevant Metadata to trace the document
	1. Date of document (to remove out of date data)
	2. Author(s) or Organizations involved
	3. Source (site or location)

A format like this would solve issue 2, all that would remain is developing a centralized tool to generate it. Considering the progress in  document understanding, I don't think this is as far fetched as it sounds.

While there exist documents that capture some of this information, like markdown, they can't capture more nuanced types of textual structure (like chapter vs header) and 
it isn't always possible to convert files to markdown.

## Example Applications of UDF

High-level context for a document would allow for filtering of a document. This can be done with current methods, so it's not really an innovation, more of just a note of 
the utility of meta data. 

A more novel, and ,IMO, important aspect is the parse tree as it would make a lot of tasks easier. For example some information is implicit if
you understand the structure of a document. For IR it might make sense to be aware of the section the text you are searching
for is in. A section might contain relevant information even if it doesn't contain relevant keywords or isn't ["semantically similar"](https://felixlabelle.github.io/2023/11/18/discussion-about-text-similarity.html). 
It might make it easier to ignore searching certain parts of a text or allow for inclusion of text that would otherwise get ignored.

When it comes to chunking text, understanding structure would allow us to chunk "in a smart way". You could avoid breaking up
one section or accidentally joining two if using a rough rule of thumb. Structure could also allow for dynamic chunking. You could scale
up and down the amount of text you are using IR over to find relevant info (some information may require who chapters (or blog posts) to convey).
If a question requires looking at entire chapters rather than subsections, (e.g., "What chapter talks about X"), a UDF could be used to
dynamically reformat structure data.

Storing structure would allow us to present information interesting to a user since the type of structure we seek to extract is a 
very human way of interacting with text. You could show someone what chapter a passage came from, or which header was associated.

Storing structure could also be useful for generative models. Providing additional context could help correct mistakes. For example 
in Speech to text, you'll often see homophones confused for one another. For example, I was watching a video on PySpark and it 
was subtitled as "Pi Spark". Had the title (or description) been included as context, it would provide context necessary to avoid the mistake.
UDF could also be used to train models and new documents or even new formats would benefit (would need to measure this to be sure) from the existing corpus. 

Life is busy at the moment, but this is something I've been thinking about to some extent for the last 4-5 years and would like to take a crack at when the opportunity arises.
Or if it already exists, let me know!