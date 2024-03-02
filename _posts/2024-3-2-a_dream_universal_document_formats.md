# Working with Documents and (annoyingly) Structured Text: The Dream of a Universal Document Format (UDF)

Writing can be hard, so here's a light post while I finish researching, testing, and writing about textual similarity methods.

Like a lot of the NLP industry since LLMs as an API have become accessible, more projects that use RAG have come up. Whether it is in the 
context of chat or a specific task, an issue I've 
come across when doing retrieval is that we often work with data that is highly varied, both in file format and presentation. We 
might have multiple types of documents with different file formats, structures, and content that all need thought and consideration
into how text is extracted and formatted (e.g. Chunking). Data can also be stored in databases which poses it's own challenges (worthy of a post in their own right).

Text extraction need to be correctly thought out or relevant information less likely to be found during the retrieval process. Extraction and text formatting currently requires effort and thought
and has a bunch of hidden caveats. I feel like until a good solution is found, RAG is likely to be less fruitful than promised for real world applications where a nice, curated corpus doesn't exist.

To that end I'd like to be able to take any file format and get back a universal document format (UDF). Something that includes 
1. Context
	1. Type of document (report, book, web page, etc..)
	2. Intent of the document
	3. Summary
2. Parse tree/graph
	1. Textual structure like headers, chapters, lists
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

High-level context for a document would allow for filtering of a document. This can be done with current methods, so it's not really an innovation, more of just a note of 
the utility of meta data. 

A more novel, and IMO important aspect is the parse tree as it would make a lot of tasks easier. For example some information is implicit if
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