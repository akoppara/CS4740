Instructions for running provided code ***PYTHON3.5***:

python main.py [sentence | perplexity | spell-check | classification] [unigram | bigram | trigram] [corpusName]

Where corpusName is one of the following:
atheism
autos
graphics
medicine
motorcycles
religion
space

Option: sentence
Args: Requires [unigram | bigram | trigram] and [corpusName]

This will generate a unigram or bigram sentence from the provided corpus name.

If unigram or bigram is not the first command line arg (or is mispelled) after main.py
or if corpusName is not one of the names listed above, the program will provide an 
error message and die gracefully.


Option: perplexity
Args: Requires [unigram | bigram | trigram] and [corpusName]

This will calculate the perplexities of each file in the classification test set individually and print out the file name and perplexity value. At the end of that print out, the average perplexity score of all the files will be displayed.

NOTE: To get all the averages side by side, replacing corpusName with all will cause the program to not output individual file perplexities but simply display all the averages

Example: python main.py perplexity bigram all


Option: spell-check
Args: Only requires [corpusName]

This will run the spell checker on the files located in [corpusName]\test_modified_docs and output the changed files into test_docs in that same folder.

Option: classification
Args: No args required

This will use the smoothed bigram model to classify all the files in the test_for_classification folder. This will not produce and console print outs, but will indicate when the process is done. The results of the classification will be output to the Kaggle-friendly format in a file called results.csv.

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