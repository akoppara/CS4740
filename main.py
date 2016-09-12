__author__ = 'Alex'
import os
import re

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
            print cleaned
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

if __name__ == '__main__':
    corpora = grab_files()
    unigram_probs, corpora_totals = calc_all_corpora_unigram(corpora)
    