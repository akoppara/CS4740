Instructions for running provided code ***PYTHON3***:

python main.py [unigram|bigram] [corpusName]

Where corpusName is one of the following:
atheism
autos
graphics
medicine
motorcycles
religion
space

This will generate a unigram or bigram sentence from the provided corpus name.

If unigram or bigram is not the first command line arg (or is mispelled) after main.py
or if corpusName is not one of the names listed above, the program will provide an 
error message and die gracefully.

Our code relies on NumPy.

Our code also assumes that the corpus data is located under the following schema:
(It is identical to how the data was provided in the cms zip file)
/
    main.py
    data_corrected/
        classification task/
            *corpora folders*

calc_all_corpora_unigram and calc_all_corpora_bigram are the functions that do the 
probability table calculations for their respective ngrams