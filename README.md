Automatic Abstract Template Generator

The project will investigate ways to extract language patterns (keywords, common phrases and structures, etc.) from about 30,000 PhD dissertation abstracts in different disciplines (Social Sciences, Arts and Humanities, Life Sciences, Physical Sciences) and develop a writing template generator that automatically produces various abstract structures for students to improve skills in writing abstracts or summaries. Solid programming skills in Python and sound knowledge in machine learning are essential. You will experience natural language processing techniques like word embedding and text generation.

Program files:
-------------------

html_process.py: Transfer HTML file to text files - train, validation, and test datasets.

tfidf.py: Calculate TF-IDF from text file

stanza.py: Create UPOS from text file  

train_run.sh, train_run.py, train.py: Train on text files and save the model.

generate_run.sh, generate_run.py, generate.py: Generate text basing on the saved model.

model.py: Model definitions.

data.py: Data reading functions.

Data directories:
-----------------

data/origin: Original text in HTML formats.

data/text_whole: Text created from HTML.

data/text: First sentences of text.

data/results: Saved models and generated text.

