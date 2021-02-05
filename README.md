Automatic Abstract Template Generator

The project will investigate ways to extract language patterns (keywords, common phrases and structures, etc.) from about 30,000 PhD dissertation abstracts in different disciplines (Social Sciences, Arts and Humanities, Life Sciences, Physical Sciences) and develop a writing template generator that automatically produces various abstract structures for students to improve skills in writing abstracts or summaries. Solid programming skills in Python and sound knowledge in machine learning are essential. You will experience natural language processing techniques like word embedding and text generation.

Program files:
-------------------

html_process.py: Transfer HTML file to text files - train and test datasets.

html_process.sh: Script to call html_process.py with parameters.

train.py: Train on text files and save the model.

train.sh: Script to call train.py with parameters.

generate.py: Generate text basing on the saved model.

generate.sh: Script to call generate.py with parameters.

model.py: Model definitions.

data.py: Data reading functions.

util.py: Public functions.

Data directories:
-----------------

data/0: Original text in HTML formats.

data/1: Text extracted from HTML.

data/2: Text with UPOS tags.

data/3: Text with UPOS tags and TFIDF tags.

data/4: Text from data/3 filtered by common words.

data/5: First sentence text extracted from data/4.

data/8: Common words directory.

data/9: Saved models and generated text.

