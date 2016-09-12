__author__ = 'Alex'
import os
import sys
import re
import copy
import numpy as np

corpus_dict = {}

def get_corpus_name(path):
    if 'train_docs' in path:
        name = re.search(r'\\\S+\\', path).group(0)
        #strip off slashes at both ends
        name = name[1:len(name)-1]
        return name
    else:
        return "test"        

def grab_files ():

    path = "data_corrected\classification task"
    for (path, dirs, files) in os.walk(path):
        if len(files):
            #Get the name of this corpus
            name = get_corpus_name(path)
            corpus = get_corpus(path, files)
            cleaned = clean_corpus(corpus)
            corpus_dict[name] = cleaned
    return corpus_dict

def clean_corpus(corpus):
    #Remove tokens that are not alphanumeric
    #Remove extraneous punctuation that doesn't cause loss of info
    #Only keep: . ' - [space] 
    allowed = ['.', "'", ' ', '-']
    cleaned_corpus = []
    for token in corpus:
        if token.isalnum():
            cleaned_corpus.append(token)
        else:
            #Remove punctuation, if string has len > 0 after keep, else discard
            #Dashes are a special case, we only want to keep them if they are part of a word
            #i.e run-down (the dash is surrounded by alphanumeric characters)
            #We don't want to keep dashes if they are the only thing in the token
            #i.e --- - -- are all invalid tokens here
            clean_token = ''
            for char in token:
                if char.isalnum() or char in allowed:
                    clean_token += char
            if len(clean_token) != 0:
                #Check for dashes
                #If removing dashes from the clean token gives it a length of 0, discard
                no_dash = clean_token.replace("-", "")
                if len(no_dash) != 0:
                    cleaned_corpus.append(clean_token)
    return cleaned_corpus

def get_corpus (path, files):
    splitted_corpus = []

    for file in files:
        file_path = path + '\\' + file
        open_file = open(file_path, 'r')
        file_string = open_file.read()
        header_ending_index = file_string.find('writes')
        #print(header_ending_index)
        if (header_ending_index != -1) :
            file_string = file_string[header_ending_index + 9:]
        elif (header_ending_index == -1) :
            header_ending_index = file_string.find('Subject')
            file_string = file_string[header_ending_index + 10:]

        #basically, if there is an 'Re : ' right after the subject
        header_ending_index = file_string.find('Re : ')
        if (header_ending_index > -1) and (header_ending_index < 5) :
            file_string = file_string[header_ending_index + 5:]

        splitted_corpus.extend(file_string.split())


    return splitted_corpus

#Get counts of individual tokens for unigram probabilities
def count_tokens(corpus):
    #corpus is an array of tokens from the data files
    counts = {}
    total = 0
    for token in corpus:
        total += 1
        if token in counts:
            counts[token] += 1
        else:
            counts[token] = 1
    #Because of no <s> tag implementation currently, a "." is used
    #as the first character of a corpus to capture the the first word as
    #being a sentence starter
    #Because of this, the unigram count is off by 1 for the "."
    counts["."] += 1
    total += 1
    return (counts, total)

def count_bigram_tokens(corpus):
    #dictonary to hold counts
    d = {}

    for i in range(len(corpus)):
        #w_n
        wordn = corpus[i]
        #If first word of corpus, add period as n-1 term
        if i == 0:
            wordn1 = "."
        else:
            wordn1 = corpus[i-1]
        if wordn1 in d:
            wd = d[wordn1]
            if wordn in wd:
                wd[wordn] += 1
            else:
                wd[wordn] = 1
        else:
            wd = {}
            wd[wordn] = 1
            d[wordn1] = wd

    return d


#Calculates the unigram probabilities for a single corpus
def calc_unigram_prob(counts, total):
    probs = {}
    for key in counts:
        count = counts[key]
        prob = float(count) / float(total)
        probs[key] = prob
    return probs

def calc_all_corpora_unigram(corpora):
    corpora_probs = {}
    corpora_totals = {}
    for key in corpora:
        unigram_counts, total_tokens = count_tokens(corpora[key])
        unigram_probs = calc_unigram_prob(unigram_counts, total_tokens)
        corpora_probs[key] = unigram_probs
        corpora_totals[key] = total_tokens

    return (corpora_probs, corpora_totals)

# bigram counts come in format with respect to example sentence
# I am the best and the worst
# { I { am : 1 } ... the { best : 1, worst : 1 } }
# { 'I' : 1/7... 'the' : 2/7 }
def calc_bigram_prob(bigram_counts, corpus):
    # need unigram counts for division later
    unigram_counts, total_tokens = count_tokens(corpus)
    # copy of bigram_counts dictionary
    bigram_probs = copy.deepcopy(bigram_counts)
    # first for loop goes through outer layer of bigram counts
    for key, value in bigram_counts.items():
        # second for loop goes through the inner layer of each outer layer
        for following_word, word_count in value.items():
            bigram_prob = float(word_count) / float(unigram_counts[key])
            bigram_probs[key][following_word] = bigram_prob
            # print (key + " " + following_word + ": ")
            # print(bigram_prob)

    # print bigram_probs
    return bigram_probs

def calc_all_corpora_bigram(corpora):
    corpora_probs = {}
    for key in corpora:
        corpus = corpora[key]
        bigram_counts = count_bigram_tokens(corpus)
        bigram_probs = calc_bigram_prob(bigram_counts, corpus)
        corpora_probs[key] = bigram_probs
    return corpora_probs

def _unigram_next_term(corpus_probs):
    #Given a set of probabilities, choose a word randomly according to those probs
    words = list(corpus_probs.keys())
    probs = list(corpus_probs.values())
    choice = np.random.choice(words, p=probs)
    return choice

def _run_unigram_gen(sentence, corpus_probs):
    next = _unigram_next_term(corpus_probs)
    if next == '.':
        if len(sentence) == 0:
            return _run_unigram_gen(sentence, corpus_probs)
        else:
            sentence += next
            return sentence
    else:
        sentence += (next + " ")
        return _run_unigram_gen(sentence, corpus_probs)

def generate_unigram_sentence(corpus, unigram_probs, start=''):
    try:
        corpus_probs = unigram_probs[corpus]
    except KeyError:
        print("ERROR: Unknown corpus")
        sys.exit(1)
    #Generation ends when period is output
    if len(start) != 0:
        start += " "
    sentence = _run_unigram_gen(start, corpus_probs)
    print(sentence)


def _bigram_next_term(prev_term, corpus_probs):
    #Get all words from corpus_probs[prev_term]
    next_word_probs = corpus_probs[prev_term]
    words = list(next_word_probs.keys())
    probs = list(next_word_probs.values())
    choice = np.random.choice(words, p=probs)
    return choice

def _run_bigram_gen(sentence, last_word, corpus_probs):
    next = _bigram_next_term(last_word, corpus_probs)
    if next == ".":
        if len(sentence) == 0:
            return _run_bigram_gen(sentence, last_word, corpus_probs)
        else:
            sentence += next
            return sentence
    else:
        sentence += (next + " ")
        return _run_bigram_gen(sentence, next, corpus_probs)

def generate_bigram_sentence(corpus, bigram_probs, start="."):
    try:
        corpus_probs = bigram_probs[corpus]
    except KeyError:
        print("ERROR: Unknown corpus")
        sys.exit(1)
    #No <s> tag currently, sentences "start" with a period (removed at end of generation)
    if start != ".":
        #Custom start, get last word of sentence
        last_word = start.split()[-1]
        sentence = _run_bigram_gen(start + " ", last_word, corpus_probs)
    else:
        sentence = _run_bigram_gen('', start, corpus_probs)
    print(sentence)

if __name__ == '__main__':
    corpora = grab_files()
    unigram_probs, corpora_totals = calc_all_corpora_unigram(corpora)
    bigram_probs = calc_all_corpora_bigram(corpora)

    args = sys.argv
    #args[1] is unigram or bigram
    #args[2] is corpus
    ngram = args[1]
    corpus = args[2]
    if ngram == "unigram":
        #Do stuff
        generate_unigram_sentence(corpus, unigram_probs)
    elif ngram == "bigram":
        #Do stuff
        generate_bigram_sentence(corpus, bigram_probs)
    else:
        print("ERROR: Unknown ngram type")
        sys.exit(1)
