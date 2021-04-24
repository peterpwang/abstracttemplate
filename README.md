Automatic Abstract Template Generator

The project will investigate ways to extract language patterns (keywords, common phrases and structures, etc.) from about 30,000 PhD dissertation abstracts in different disciplines (Social Sciences, Arts and Humanities, Life Sciences, Physical Sciences) and develop a writing template generator that automatically produces various abstract structures for students to improve skills in writing abstracts or summaries. Solid programming skills in Python and sound knowledge in machine learning are essential. You will experience natural language processing techniques like word embedding and text generation.


Program files:
-------------------

html_process.py: Transfer HTML file to text files - train and test datasets.

html_process.sh: Script to call html_process.py with parameters.

train.py: Train on text files and save the model.

train.sh: Script to call train.py with parameters.

generate.py: Generate text basing on the saved model.

generate.sh: Script to call generate.py with parameters to generate conditional (interactive) text.

generate_uncond.sh: Script to call generate.py with parameters to generate unconditional text.

calculate_perplexity.py: Calculate perplexity of generated text.

util.py: Public functions.


Data directories:
-----------------

data/0: Original text in HTML formats.

data/1: Text extracted from HTML.

data/8: Common words directory.

data/9: Saved models and generated text.


pplm directories:
-----------------

pplm: PPLM source code from https://github.com/uber-research/PPLM/tree/master/paper_code. (Improved output for interactive text generation)


Data process, training and text generation instructions
-------------------------------------------------------

1. Save html files of theses in HTML format into data/0/.

2. Run html_process.sh and extract text from theses and save into data/1/.

3. Install GPT-2 from https://github.com/nshepperd/gpt-2.git and run train.sh to create customised GPT-2 model from data/1/ into data/9/gpt2-theses/.

4. Install PPLM and run generate.sh or generate_cond.sh to generate text based on customised GPT-2 model in data/9/gpt2-theses/ and PPLM model.

5. Run python ./calculate_perplexity.py generated.txt to calculate perplexity of generated text stored in "generated.txt".


