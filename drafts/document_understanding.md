# Working with Documents and (annoyingly) Structured Text

An unspoken truth of NLP is that text data is rarely just nice and ready for use. Besides noise, there are other issues like how and where text is stored. Text data is often 
written for people and formatted in ways for which models are not trained. Text often uses lists or visual cues to indicate structure. As far as I know, current models cannot easily work with that unless
the structure is indicated using markdown or some other programming language to 

While there have been improvements in tooling to get data out of documents, when working with data formatted for humans there are 
challenges for NLP. It would be nice if I could chuck data into a model without any preprocessing or reformatting. Sadly we're just not there yet.


This blog will cover 

1. Background on working with structured data
2. Some solutions using existing tools
3. What I'd like to see in the future (The Dream)

## Utility of Document Understanding

### Information Retrieval

Better able to understand references. Use of implicit information

### Ability to fully leverage multimodal inputs

References between modalities and types of information

tables, images, references,

## Current Technologies

A lot of the approaches are currently file type specific, for that reason I will break down current approaches using that

### PDFs

#### PDF parsers

#### OCR

Current tooling has come a long way. I remember working with PDFs 5 years ago was like being in the stone ages. Now there exist a plethora
of quick and efficient OCR tools for text extraction.

### Word Documents

#### .docx

#### .doc

### Formatted Text

#### MD

#### HTML

### Free format

#### Lists

One format of data that's common is lists. The example I'm most familiar with is regulations. These have a very specific structure, although
the names and depth of this structure may vary widely between regulators.

#### Epub


##  The Dream

I'd like to be able to take any file format and get back a universal data format. Something that includes 
1. Type of document (report, book, web page, etc..)
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

The parse tree might seem odd, but it would make a lot of tasks easier. For example some information is implicit if
you understand the structure of a document. For IR it might make sense to be aware of the section the text you are searching
for is in. It might make it easier to ignore searching certain parts of a text. 

When it comes to chunking text, understanding structure would allow us to chunk "in a smart way". You could avoid breaking up
one section or accidently joining two if using a rough rule of thumb. Structure could also allow for dynamic chunking. You could scale
up and down the amount of text you are using IR over to find relevant info (some information may require who chapters (or blog posts) to convey).

Storing structure would allow us to present information interesting to a user since the type of structure we seek to extract is a 
very human way of interacting with text. You could show someone what chapter a passage came from, or which header was associated.

