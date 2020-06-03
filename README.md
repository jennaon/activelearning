EE364 Final Project on Active Learning.

References:
[Main reference, Ralaivola & Louche 2015](https://arxiv.org/abs/1508.02986)

[Secondary reference on text classification, Tong & Koller 2001](http://www.jmlr.org/papers/volume2/tong01a/tong01a.pdf)

Setup your environment:

`conda env create -f environment.yml`

You will need to set up a few nltk packages locally. Start your `conda` enviornment
`conda activate cvxpy`

and start Python Interpreter:
`python`

Download necessary packages:
`improt nltk
nltk.download('reuters', 'punkt', 'stopwords')``

If successful, you'll get "true" message.

Next, you will need to create pickled data. To do this, run
`python preprocess.py`

You should be able to see tokenization & pickling successful message at the end.

Now you're ready to roll with `estimator.py` code!

DLU 6/2/20 JL
