# The Absolute State: Working with Documents and (annoyingly) Structured Text

An unspoken truth of NLP is that text data is rarely just nice and ready for use. Besides noise, there are other issues like how and where text is stored. Text data is often 
written for people and formatted in ways for which models are not trained. Text often uses lists or visual cues to indicate structure. As far as I know, current models cannot easily work with that unless
the structure is indicated using markdown or some other programming language to 

While there have been improvements in tooling to get data out of documents, when working with data formatted for humans there are 
challenges for NLP. It would be nice if I could chuck data into a model without any preprocessing or reformatting. Sadly we're just not there yet.

## Caveats

There is research in the space, but frankly I'm not familiar with it. Here's a link with relevant info.
https://github.com/tstanislawek/awesome-document-understanding?tab=readme-ov-file#introduction
Rather I will focus on Python tools
I will only mention
python tools, but the general concepts below are langauge independent should your needs differ


## Current Approach

In my experience people rarely consider consider when working with text. The goal is often to just "get the text out". 
Often getting the text out is dependent on file format, document type, and the end use case.
	
Due to how file format specific existing text extraction is, the following sections are broken down by file format.

### PDFs

PDFs are a hot mess to work with in my experience, especially if you want to be able to get working text out of them. Document level text extraction 
tends to be simple, but being able to pull finer granularities of data in a meaningful way is hard, since most extraction methods can't distinguish between relevant
and irrelevant text (thing figure captions). I would approach PDFs with caution. At a high level I would say there are 3 categories of methods
to extract text from PDFs. There are 
1. Parsers
2. OCR Tools
3. Segmenter

#### Parsers

I'm not a PDF format expert, but in my understanding they are essentially a mix of different objects that layout visuals. As such they can be parsed.

Tools like PDFMiner, pypdf, PyMuPDF can achieve this

Issues include multiple versions. Parsed text can be a mess because how text is structured is variable. If text is an image it can't be extracted. Hard to tell 
sections/types of text from one another. Physical gaps and boundaries

#### OCR Tools

OCR tools don't work directly with the PDFs data, but rather treat it as an image and returns the text contained.

Examples include Surya  [Tesseract](https://github.com/tesseract-ocr/tesseract)

AFAICT OCR tools don't necessarily understand structure and may struggle with broken up text. That having been said, they
won't be confused by any issues in how text is formatted

#### Segmenters

The final type of PDF parser I'm aware of is segmenters. Essentially they create
bounding boxes (identify regions). This makes operations like ignoring certain 
elements easier. OCR may be required to extract text from regions and some clean up may be required.


Examples for this are sparser. The most public example I'm aware of is [Semantic Scholar](https://www.semanticscholar.org/).
AllenAI has some open source work that is likely the engine behind Semantic Scholar [v1]() [v2]()

An alternative would be to train open source segmentation models like YoloV5 and create your own segmentation data (classes and bounding boxes).

### Word Suite

By word suite I mean excel, powerpoint, word documents. There exist open formats as well such as ...

Word suite documents are (as of 3/2/2024) zip archives and with XML denoting how the files in this archive are combined together to for the document. This is denoted by the 
"x" at the end of the file extension. For example .xlsx, .docx, .pptx. There exist legacy formats that do not follow these conventions. Rather than list
every word suite format I will just cover tools to deal with both

#### Zip Format data

#### Legacy formats

### Structured data

#### HTML/XML

Beautiful soup

#### Markdown



#### Custom parsers: Lists or other file formats

#### CSV

### Other document formats I've worked with

#### epub

## Putting together a text extraction program 

Considerations include

1. Format of text output
	1. Is it chunked 
	2. What size of chunks are desired
	3. Do chunks need to be complete
	4. Do chunks overlap
2. What text do you want/is there anything that could cause issues?
	1. Figures or text explaining data
	2. Tables and values within them
	3. Normal Text
	4. Tables of contents
3. Corpus level considerations
	1. Mix of file formats
	2. Amount of data
	3. Consistency of formats
	4. Number of documents
4. Performance
	1. Hardware available
		1. CPU vs GPU
		2. RAM requirements
	2. Time requirements
	3. Output space requirements

### Evaluation

Sampling data at the end

Benchmarking for performance

Slowly scaling up

##  Conclusion

Currently document extraction requires quite a bit of planning and awareness of the down stream use case.
You need to know what types of documents and what your text needs to look like.

Some amongst you may say, "We now have multimodal models, no need to have an explicit text extraction step any more".
While I think having large multimodal base models might seem like, and in someone ways is, a step towards text extraction, it leaves things to be desired.
Context length and hallucinations are real issues when trying to get quick and high fidelity outputs. I think the, more immediate future, is better document understanding 
methods and universal document formats (UDF).