__author__ = 'Alex'
import os
import sys
import re
import copy
import numpy as np
import math
import operator
import json
import csv

corpus_dict = {}

UNK_TOKEN = "<unk>"
N1_N0 = {}
unigram_threshold = 100000
bigram_threshold = 100000

def get_corpus_name(path):
    if 'train_docs' in path:
        name = re.search(r'\\[0-9a-zA-Z]+\\', path).group(0)
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
            if name != "test":
                corpus = get_corpus(path, files)
                cleaned = add_unk_token(clean_corpus(corpus))
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

def add_unk_token(corpus):
    counts, totals = count_tokens(corpus)
    word_counter = 0
    for word in corpus:
        if (counts[word] == 1):
            corpus[word_counter] = UNK_TOKEN
        word_counter += 1
    return corpus

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

def get_test_corpus():
    test_files = {}
    path = "data_corrected\classification task\\test_for_classification"
    for (path, dirs, files) in os.walk(path):
        for file in files:
            file_path = path + "\\" + file
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
            test_files[file] = clean_corpus(file_string.split())
    return test_files

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

def count_trigram_tokens(corpus):
    d = {}
    for i in range (len(corpus)):
        wordn = corpus[i]

        if (i == 0):
            wordn1 = "."
            wordn2 = "."
        elif (i == 1):
            wordn1= corpus[i-1]
            wordn2 = "."
        else:
            wordn1 = corpus[i-1]
            wordn2 = corpus[i-2]

        if wordn2 in d:
            wd2 = d[wordn2]
            if wordn1 in wd2:
                wd1 = wd2[wordn1]
                if wordn in wd1:
                    wd1[wordn] += 1
                else:
                    wd1[wordn] = 1
            else:
                wd1 = {}
                wd1[wordn] = 1
                wd2[wordn1] = wd1
        else:
            wd2 = {}
            wd1 = {}
            wd1[wordn] = 1
            wd2[wordn1] = wd1
            d[wordn2] = wd2

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

        #_,unigram_counts = calc_all_corpora_unigram(corpora)
        #vocab_size = unigram_counts[key]
        #calc_gt_probability(bigram_counts, vocab_size)
    return corpora_probs

def calc_trigram_prob(trigram_counts, corpus):
    # need unigram counts for division later
    unigram_counts, total_tokens = count_tokens(corpus)
    bigram_counts = count_bigram_tokens(corpus)
    # copy of bigram_counts dictionary
    trigram_probs = copy.deepcopy(trigram_counts)
    # first for loop goes through outer layer of trigram counts
    for first_word, second_words in trigram_counts.items():
        # second for loop goes through the inner layer of each outer layer
        for second_word, third_words in second_words.items():
            for third_word, word_count in third_words.items():
                #print(bigram_counts[first_word][second_word])
                trigram_prob = float(word_count) / float(bigram_counts[first_word][second_word])
                trigram_probs[first_word][second_word][third_word] = trigram_prob

    #print (trigram_probs['.']['.'])
    return trigram_probs

def calc_all_corpora_trigram(corpora):
    corpora_probs = {}
    for key in corpora:
        corpus = corpora[key]
        trigram_counts = count_trigram_tokens(corpus)
        trigram_probs = calc_trigram_prob(trigram_counts, corpus)
        corpora_probs[key] = trigram_probs
        #print(trigram_probs)

        #_,unigram_counts = calc_all_corpora_unigram(corpora)
        #vocab_size = unigram_counts[key]
        #calc_gt_probability(bigram_counts, vocab_size)

    return corpora_probs

def _unigram_next_term(corpus_probs):
    #Given a set of probabilities, choose a word randomly according to those probs
    words = list(corpus_probs.keys())
    probs = list(corpus_probs.values())
    choice = np.random.choice(words, p=probs)
    return choice

def _run_unigram_gen(sentence, corpus_probs, limit=15):
    next = _unigram_next_term(corpus_probs)
    if next == '.':
        if len(sentence) == 0:
            return _run_unigram_gen(sentence, corpus_probs)
        else:
            sentence += next
            return sentence
    else:
        sentence += (next + " ")
        if len(sentence.split()) < limit:
            return _run_unigram_gen(sentence, corpus_probs)
        else:
            return sentence

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


def _trigram_next_term(last_term, word_before_last_word, corpus_probs):
    #Get all words from corpus_probs[prev_term]
    next_word_probs = corpus_probs[word_before_last_word][last_term]
    words = list(next_word_probs.keys())
    probs = list(next_word_probs.values())
    # print(sum(probs))
    diff = sum(probs) - 1
    if (sum(probs) != 1):
        if '.' in corpus_probs[word_before_last_word][last_term]:
            corpus_probs[word_before_last_word][last_term]['.'] -= diff
            words = list(next_word_probs.keys())
            probs = list(next_word_probs.values())
    choice = np.random.choice(words, p=probs)
    return choice

def _run_trigram_gen(sentence, last_word, word_before_last_word, corpus_probs):
    next = _trigram_next_term(last_word, word_before_last_word, corpus_probs)
    #print(next)
    if next == ".":
        if len(sentence) == 0:
            return _run_trigram_gen(sentence, last_word, word_before_last_word, corpus_probs)
        else:
            sentence += next
            return sentence
    else:
        sentence += (next + " ")
        return _run_trigram_gen(sentence, next, last_word, corpus_probs)

def generate_trigram_sentence(corpus, trigram_probs, start="."):
    try:
        corpus_probs = trigram_probs[corpus]
    except KeyError:
        print("ERROR: Unknown corpus")
        sys.exit(1)
    #No <s> tag currently, sentences "start" with a period (removed at end of generation)
    if start != ".":
        #Custom start, get last word of sentence
        last_word = start.split()[-1]
        word_before_last_word = start.split()[-2]
        sentence = _run_trigram_gen(start + " ", last_word, word_before_last_word, corpus_probs)
    else:
        sentence = _run_trigram_gen('', start, start, corpus_probs)
    print(sentence)



def calc_gt_all_corpora_bigram (corpora):
    corpora_probs = {}
    corpora_totals = {}
    for key, corpus in corpora.items():
        corpus_probs = {}
        bigram_counts = count_bigram_tokens(corpus)
        unigram_counts, total_tokens = count_tokens(corpus)
        p_star_values, total_c_star, c_star_values = calc_bigram_gt_prob(corpus)
        #print(p_star_values)

        for word, following in bigram_counts.items():
            p_sum = 0
            sum_of_counts = 0
            for following_word, count in following.items():
                sum_of_counts += c_star_values[count]

            following_probs = {}
            for following_word, count in following.items():
                prob = c_star_values[count] / sum_of_counts
                following_probs[following_word] = prob
                p_sum += following_probs[following_word]
            corpus_probs[word] = following_probs

        corpora_probs[key] = corpus_probs
        corpora_totals[key] = total_c_star

    return (corpora_probs)

#adjust threshold if necessary
def calc_bigram_gt_prob (corpus):
    bigram_freq, total_possible = calc_bigram_freq(corpus)
    c_star_values = {}
    p_star_values = {}
    p_star_total = 0

    for freq, freq_of_freq in bigram_freq.items():
        first_or_not_empty = (bigram_freq[freq] != 0) or (freq == 0) 
        if ((first_or_not_empty) and (freq < len(bigram_freq)-1)):
            if (bigram_freq[freq + 1] != 0):                 
                if (freq != 0):
                    c_star = (freq + 1) * ( ( bigram_freq[freq + 1] ) / ( bigram_freq[freq] ) )
                    c_star_values[freq] = c_star
                else:
                    c_star = bigram_freq[freq + 1]
                    c_star_values[freq] = c_star
            else:
                    c_star = bigram_freq[freq]
                    c_star_values[freq] = c_star
        else:
            c_star = bigram_freq[freq]
            c_star_values[freq] = c_star

    total_possible = 0
    for freq, c_star in c_star_values.items():
        total_possible += c_star


    #print(c_star_values) 
    count = 0
    for freq, c_star in c_star_values.items():
        p_star = c_star / total_possible
        if (count < bigram_threshold):
            p_star_values[freq] = p_star
            count += 1
        elif (count == bigram_threshold):
            p_star_values[freq] = 0
            count += 1
    count = 0
    for freq, p_star in p_star_values.items():
        if (count == 0):
            p_star_total += p_star 
        elif (count <= bigram_threshold):
            p_star_total += p_star * freq
            count += 1

    return (p_star_values, total_possible, c_star_values)



#bigram_counts: if given a sentence "I jumped over the fence and I fell"
#the bigram counts look like { 'I' : { 'jumped' : 1 , 'fell' : 1 } ... }
#the goal of this function is to count the frequency of each frequency of bigram happening
#ex. there were 3 bigrams that occurred 1 time, 2 that occurred 2 times, 1 that occurred 3 times, and 10 that occurred 0 times
#to calculate the bigrams that occurred 0 times, it is (V^2 - bigrams that have occurred) where V is the vocabulary
def calc_bigram_freq (corpus):
    bigram_counts = count_bigram_tokens(corpus)
    unigram_counts, vocab_size = count_tokens(corpus)
    total = vocab_size ** 2

    #print(bigram_counts)
    bigram_freq = {}
    max_value = 0
    sum_of_freq = 0
    for key, value in bigram_counts.items():
        for word, frequency in value.items():
            if frequency > max_value:
                max_value = frequency
    for x in range(0, max_value+1):
        bigram_freq[x] = 0
    for key, value in bigram_counts.items():
        for word, frequency in value.items():
            bigram_freq[frequency] += 1
    for key, value in bigram_freq.items():
        sum_of_freq += value
    #print(vocab_size)
    freq_for_0 = (vocab_size ** 2) - sum_of_freq
    #print(sum_of_freq)
    #print(vocab_size)
    bigram_freq[0] = freq_for_0
    #print(bigram_freq)
    return (bigram_freq, total)


def calc_gt_all_corpora_unigram (corpora):
    corpora_probs = {}
    corpora_totals = {}

    for key, corpus in corpora.items():
        corpus_probs = {}
        unigram_counts, total_tokens = count_tokens(corpus)
        p_star_values, total_c_star = calc_unigram_gt_prob(key, corpus)
        p_sum = 0

        freq_of_freqs = {}
        for word, count in unigram_counts.items():
            if count in freq_of_freqs:
                freq_of_freqs[count] += 1
            else:
                freq_of_freqs[count] = 1
        for word, count in unigram_counts.items():

            prob = p_star_values[count] / freq_of_freqs[count]
            corpus_probs[word] = prob
            p_sum += corpus_probs[word]
        #print(corpus_probs)
        # print(p_sum)
        corpora_probs[key] = corpus_probs
        corpora_totals[key] = total_c_star

    #print(corpora_probs)
    return (corpora_probs, corpora_totals)


def calc_unigram_gt_prob (name, corpus):
    unigram_freq, vocab_size = calc_unigram_freq(name, corpus)
    c_star_values = {}
    p_star_values = {}
    p_star_total = 0

    for freq, freq_of_freq in unigram_freq.items():
        first_or_not_empty = (unigram_freq[freq] != 0) or (freq == 0) 
        if ((first_or_not_empty) and (freq < len(unigram_freq)-1)):
            if (unigram_freq[freq + 1] != 0):                 
                if (freq != 0):
                    c_star = (freq + 1) * ( ( unigram_freq[freq + 1] ) / ( unigram_freq[freq] ) )
                    c_star_values[freq] = c_star
                else:
                    c_star = unigram_freq[freq + 1] / vocab_size
                    c_star_values[freq] = c_star
            else:
                    c_star = unigram_freq[freq]
                    c_star_values[freq] = c_star
        else:
            c_star = unigram_freq[freq]
            c_star_values[freq] = c_star

    vocab_size = 0
    for freq, c_star in c_star_values.items():
        vocab_size += c_star


    #print(c_star_values) 
    count = 0
    for freq, c_star in c_star_values.items():
        p_star = c_star / vocab_size
        if (count < unigram_threshold):
            p_star_values[freq] = p_star
            count += 1
        elif (count == unigram_threshold):
            p_star_values[freq] = 0
            count += 1
    count = 0
    for freq, p_star in p_star_values.items():
        if (count == 0):
            p_star_total += p_star 
        elif (count <= unigram_threshold):
            p_star_total += p_star * freq
            count += 1

    #print(p_star_values)
    #print(p_star_total)
    return (p_star_values, vocab_size)

def calc_unigram_freq (name, corpus):
    counts, total = count_tokens(corpus)
    frequency = {}
    max_value = 0
    for word, freq in counts.items():
        if (freq > max_value):
            max_value = freq
    for x in range(0, max_value+1):
        frequency[x] = 0
    for word, freq in counts.items():
        frequency[freq] += 1

    #Because we replace all words with occurrance 1 with <unk>,
    # frequency[1] would be 0, so frequency[1] is actually count of unks
    frequency[1] = counts[UNK_TOKEN]
    # frequency[counts[UNK_TOKEN]] -= 1
    # print(frequency[1])
    N1_N0[name] = frequency[1] / total
    frequency[1] = 0
    return frequency, total

 
def _unigram_perplexity(unigram_probs, test_file):
    #Calculate perplexity of test_set given the unigram_probs of the corpus
    #test_file will be an array of tokens for a given file
    #calc sum of log probs
    summation = 0.0
    N = len(test_file)
    for i, word in enumerate(test_file):
        try:
            prob = unigram_probs[word]
        except KeyError:
            prob = unigram_probs[UNK_TOKEN]
        summation += (-1) * (math.log(prob))
    #multiply by 1/N
    result = (1.0 / N) * summation
    #answer is e^result
    return math.exp(result)

def calc_unigram_perplexity(corpora, unigram_probs, test_corpus):
    #For each corpus in corpora, compute perplexity of every file in test_corpus
    perplexity_data = {}
    for corpus in corpora:
        corpus_data = {}
        for filename in test_corpus:
            test_file = test_corpus[filename]
            perplexity = _unigram_perplexity(unigram_probs[corpus], test_file)
            corpus_data[filename] = perplexity
        perplexity_data[corpus] = corpus_data
    return perplexity_data


def _bigram_perplexity(corpus, bigram_probs, test_file, unigram_counts):
    summation = 0.0
    N = len(test_file)
    for i, word in enumerate(test_file):
        if i==0:
            prev_word = "."
            if "." in bigram_probs:
                outer_prob = bigram_probs["."]
            else:
                outer_prob = bigram_probs[UNK_TOKEN]

            if word in outer_prob:
                prob = outer_prob[word]
            elif UNK_TOKEN in outer_prob:
                prob = outer_prob[UNK_TOKEN]
            else:
                prob = N1_N0[corpus] /unigram_counts["."]
        else:
            prev_word = test_file[i-1]
            if prev_word in bigram_probs:
                outer_prob = bigram_probs[prev_word]
            else:
                prev_word = UNK_TOKEN
                outer_prob = bigram_probs[UNK_TOKEN]

            if word in outer_prob:
                prob = outer_prob[word]
            elif UNK_TOKEN in outer_prob:
                prob = outer_prob[UNK_TOKEN]
            else:
                prob = N1_N0[corpus] / unigram_counts[prev_word]

        # print "WORDS: " + prev_word + ", " + word + " PROB: " + str(prob)

        summation += (-1) * (math.log(prob))
    #multiply by 1/N
    result = (1.0 / N) * summation
    #answer is e^result
    return math.exp(result) 


def calc_bigram_perplexity(corpora, bigram_probs, test_corpus):
    perplexity_data = {}
    for corpus in corpora:
        corpus_data = {}
        for filename in test_corpus:
            test_file = test_corpus[filename]
            unigram_counts, _ = count_tokens(corpora[corpus])
            perplexity = _bigram_perplexity(corpus, bigram_probs[corpus], test_file, unigram_counts)
            corpus_data[filename] = perplexity
        perplexity_data[corpus] = corpus_data
    return perplexity_data


def _trigram_perplexity(corpus, trigram_probs, test_file, unigram_counts):
    summation = 0.0
    N = len(test_file)
    for i, word in enumerate(test_file):
        prob = 0.0
        if i == 0:
            #Don't have enough to compute a probability
            continue
        elif i == 1:
            n_2 = "."
            n_1 = test_file[i-1]
        else:
            n_2 = test_file[i-2]
            n_1 = test_file[i-1]

        n = test_file[i]

        if n_2 in trigram_probs:
            outermost_prob = trigram_probs[n_2]
        else:
            outermost_prob = trigram_probs[UNK_TOKEN]
            n_2 = UNK_TOKEN

        if n_1 in outermost_prob:
            inner_prob = outermost_prob[n_1]
        elif UNK_TOKEN in outermost_prob:
            n_1 = UNK_TOKEN
            inner_prob = outermost_prob[UNK_TOKEN]
        else:
            prob = N1_N0[corpus] / (N1_N0[corpus] + unigram_counts[n_2])

        #At this point, if prob is not set, then keep going else, stop
        if prob == 0:
            if n in inner_prob:
                prob = inner_prob[n]
            elif UNK_TOKEN in inner_prob:
                prob = inner_prob[UNK_TOKEN]
            else:
                prob = N1_N0[corpus] / (unigram_counts[n_1] + unigram_counts[n_2])


        summation += (-1) * (math.log(prob))
    #multiply by 1/N
    result = (1.0 / N) * summation
    #answer is e^result
    return math.exp(result) 

def calc_trigram_perplexity(corpora, trigram_probs, test_corpus):
    perplexity_data = {}
    for corpus in corpora:
        corpus_data = {}
        for filename in test_corpus:
            test_file = test_corpus[filename]
            unigram_counts, _ = count_tokens(corpora[corpus])
            perplexity = _trigram_perplexity(corpus, trigram_probs[corpus], test_file, unigram_counts)
            corpus_data[filename] = perplexity
        perplexity_data[corpus] = corpus_data
    return perplexity_data


#Topic Classification code
#For each corpus, calculate perplexity of each file
#Results dict will have form:
#   {file_i.txt : {corpus1: perp1, corpus2: perp2 ...}}

def topic_classification():
    corpora = grab_files()
    #Need to generate N1_N0
    calc_gt_all_corpora_unigram(corpora)
    test_corpus = get_test_corpus()
    bigram_probs = calc_gt_all_corpora_bigram(corpora)
    perplexity_data = calc_bigram_perplexity(corpora, bigram_probs, test_corpus)
    
    classification_data = {}

    for file_name in test_corpus:
        classification_data[file_name] = {}

    #perplexity_data has the form:
    #   {corpus1: {file_i.txt: perp_i, file_j.txt: perp_j}}

    for corpus, file_perp in perplexity_data.items():
        for file_name, perp_score in file_perp.items():
            classification_data[file_name][corpus] = perp_score

    classification_key = {
        "atheism" : 0,
        "autos" : 1,
        "graphics" : 2,
        "medicine" : 3,
        "motorcycles" : 4,
        "religion" : 5,
        "space" : 6
    }

    #Final output should be a dictionary like:
    #   {file_i.txt : corpus_key}

    results = {}

    for file_name in classification_data:
        perplexities = classification_data[file_name]
        topic = min(perplexities.items(), key=operator.itemgetter(1))[0]
        # print(file_name + ", " + topic)
        results[file_name] = classification_key[topic]

    # json.dump(results, open("classification_results.json", "w"))

    #Need to export to csv
    export_classification_results(results)


def export_classification_results(results):
    with open("results.csv", "w") as csvfile:
        field_names = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n', fieldnames=field_names)
        writer.writeheader()
        lines = []
        for file_name, prediction in results.items():
            lines.append({"Id": file_name, "Prediction" : prediction})
        for line in lines:
            writer.writerow(line)
    print("DONE -- output is in results.csv")


#Spell checking code
#Create bigram model of text in train_docs for each topic
#Parse confusion set into dictionary
#For testing:
#   Parse each file in train_modified_docs and copy non-confusion words to new "clean file"
#   Whenever a confusion set word is encountered, use bigram model to predict next word
#   Given word before confusion word, which word is more likely to be there?

def grab_spelling_files():

    path = "data_corrected\spell_checking_task"
    for (path, dirs, files) in os.walk(path):
        if len(files):
            #Get the name of this corpus
            name = get_corpus_name(path)
            if "train_docs" in path:
                corpus = []
                for file in files:
                    file_path = path + '\\' + file
                    open_file = open(file_path, 'r')
                    file_string = open_file.read()
                    corpus.extend(file_string.split())
                # corpus = get_corpus(path, files)
                # cleaned = clean_corpus(corpus)
                corpus_dict[name] = add_unk_token(corpus)
    return corpus_dict

def parse_confusion_set():
    path = "data_corrected\spell_checking_task"
    file_name = "confusion_set.txt"
    file_path = path + "\\" + file_name
    open_file = open(file_path, 'r')
    file_string = open_file.read()

    pairs = file_string.split("\n")

    confusion_set = {}

    for pair in pairs:
        #pair is a string like "word1 word2"
        #create a dictionary where for each pair there are two entries of the form
        #dict[word1] = word2
        #dict[word2] = word1

        words = pair.split()
        word1 = words[0]
        word2 = words[1]

        if word1 in confusion_set:
            confusion_set[word1].append(word2)
        else:
            confusion_set[word1] = [word2]

        if word2 in confusion_set:
            confusion_set[word2].append(word1)
        else:
            confusion_set[word2] = [word1]
        # confusion_set[word1] = word2
        # confusion_set[word2] = word1


    return confusion_set

def spell_check(corpus, tokens):

    #Generate bigram model of this corpus
    unigram_counts, _ = count_tokens(tokens)
    spelling_corpora = grab_spelling_files()

    spelling_bigrams = calc_all_corpora_bigram(spelling_corpora)

    bigram_probs = spelling_bigrams[corpus]

    #Parse the confusion set
    confusion_set = parse_confusion_set()


    path = "data_corrected\spell_checking_task\%s\\test_modified_docs" % corpus

    clean_path = "data_corrected\spell_checking_task\%s\\test_docs" % corpus

    if not os.path.exists(clean_path):
        os.mkdir(clean_path)

    files = os.listdir(path)

    #For each file in data_corrected\spell_checking_task\{corpus}\train_modified_docs:
    for file_name in files:
    #   Read the file into an array
        file_path = path + '\\' + file_name
        open_file = open(file_path, 'r')
        file_string = open_file.read()
        file_tokens = file_string.split()
    #   Clean out junk from tokens
        # file_tokens = clean_corpus(file_tokens)
    #   For each token in the array:
        for i, token in enumerate(file_tokens):
    #       If it is in the confusion set:
            if token in confusion_set:
    #           Get the word that came before this
                if i == 0:
                    #first word, use "." as prev_token
                    prev_token = "."
                else:
                    prev_token = file_tokens[i-1]
    #           Pick the word that is more likely to be there (must be a confusion word)
                
                if prev_token in bigram_probs:
                    next_token_probs = bigram_probs[prev_token]
                else:
                    prev_token = UNK_TOKEN
                    next_token_probs = bigram_probs[UNK_TOKEN]

                token_probs = {}

                if token in next_token_probs:
                    curr_correct = next_token_probs[token]
                elif UNK_TOKEN in next_token_probs:
                    curr_correct = next_token_probs[UNK_TOKEN]
                else:
                    #TODO: Uncomment me when smoothing done
                    # curr_correct = c* / count(prev_token)
                    try:
                        curr_correct = N1_N0[corpus] / unigram_counts[prev_token]
                    except KeyError:
                        curr_correct = N1_N0[corpus] / unigram_counts[UNK_TOKEN]

                token_probs[token] = curr_correct

                other_options = confusion_set[token]
                #for each other option, get the prob
                for other_token in other_options:
                    if other_token in next_token_probs:
                        other_correct = next_token_probs[other_token]
                    elif UNK_TOKEN in next_token_probs:
                        other_token = UNK_TOKEN
                        other_correct = next_token_probs[UNK_TOKEN]
                    else:
                        #TODO: Uncomment me when smoothing done
                        # other_correct = c* / count(prev_word)
                        try:
                            other_correct = N1_N0[corpus] / unigram_counts[prev_token]
                        except KeyError:
                            other_correct = N1_N0[corpus] / unigram_counts[UNK_TOKEN]

                    token_probs[other_token] = other_correct

                correct_token = max(token_probs.items(), key=operator.itemgetter(1))[0]


                # print ("Previous Token: ", prev_token)
                # print ("Current Token: ", token)
                # print ("Probs: ", token_probs)
                # print ("Correct Token: ", correct_token)

                #check if probs are all 0
                correct_prob = token_probs[correct_token]

                if correct_prob != 0.0:
                    file_tokens[i] = correct_token

    #       If is not in confusion set:
    #           Do nothing, leave as is in file_tokens
        file_string_out = " ".join(file_tokens)
        with open(clean_path+ "\\" +file_name, 'w') as f:
            f.write(file_string_out)

    #   Recombine cleaned tokens array into string

    #   Save as file_name in data_corrected\spell_checking_task\{corpus}\train_modified_docs_TEST


def handle_sentence_generation(ngram, corpus):
    corpora = grab_files()
    if ngram == "unigram":
        #Non smoothed
        # unigram_probs, corpora_totals = calc_all_corpora_unigram(corpora)
        #Smoothed
        unigram_probs, _ = calc_gt_all_corpora_unigram(corpora)
        generate_unigram_sentence(corpus, unigram_probs)
    elif ngram == "bigram":
        #Non smoothed
        # bigram_probs = calc_all_corpora_bigram(corpora)
        #Smoothed
        bigram_probs = calc_gt_all_corpora_bigram(corpora)
        generate_bigram_sentence(corpus, bigram_probs)
    elif ngram == "trigram":
        #unsmoothed
        trigram_probs = calc_all_corpora_trigram(corpora)
        # print(trigram_probs)
        generate_trigram_sentence(corpus, trigram_probs)
    else:
        print("ERROR: Unknown ngram type")
        sys.exit(1)

def _pretty_print_perplexity(data):
    print("PERPLEXITY")
    for file_name, perplexity in data.items():
        print(file_name + ": " + str(perplexity))

def _print_perplexity_avg(corpus, data):
    values = list(data.values())
    total = sum(values)
    avg = float(total) / len(values)
    print(corpus + ": " + str(avg))
def handle_perplexity_calculation(ngram, corpus):
    corpora = grab_files()
    #Need to generate N1_N0
    calc_gt_all_corpora_unigram(corpora)
    test_corpus = get_test_corpus()
    if ngram == "unigram":
        #Non smoothed
        # unigram_probs, corpora_totals = calc_all_corpora_unigram(corpora)
        unigram_probs, _ = calc_gt_all_corpora_unigram(corpora)
        perplexity_data = calc_unigram_perplexity(corpora, unigram_probs, test_corpus)
    elif ngram == "bigram":
        #Non smoothed
        # bigram_probs = calc_all_corpora_bigram(corpora)
        #Smoothed
        bigram_probs = calc_gt_all_corpora_bigram(corpora)
        perplexity_data = calc_bigram_perplexity(corpora, bigram_probs, test_corpus)
    elif ngram == "trigram":
        trigram_probs = calc_all_corpora_trigram(corpora)
        perplexity_data = calc_trigram_perplexity(corpora, trigram_probs, test_corpus)
    else:
        print("ERROR: Unknown ngram type")
        sys.exit(1)

    if corpus != "all":
        try:
            data = perplexity_data[corpus]
        except KeyError:
            print("ERROR: Unknown corpus")
            sys.exit(1)
        _pretty_print_perplexity(data)
        print("======Average======")
        _print_perplexity_avg(corpus, data)
    else:
        # for corpus in corpora:
        #     _pretty_print_perplexity(perplexity_data[corpus])
        #Show averages
        print("Averages for " + ngram + ":")
        for corpus in corpora:
            data = perplexity_data[corpus]
            _print_perplexity_avg(corpus, data)
    sys.exit(0)

def handle_spell_checker(corpus):
    corpora = grab_files()
    calc_gt_all_corpora_unigram(corpora)
    try:
        dummy = corpora[corpus]
    except KeyError:
        print("ERROR: Unknown corpus")
        sys.exit(1)
    spell_check(corpus, corpora[corpus])
    print("DONE -- " + corpus)

if __name__ == '__main__':
    # corpora = grab_files()
    # unigram_probs, corpora_totals = calc_all_corpora_unigram(corpora)
    # bigram_probs = calc_all_corpora_bigram(corpora)

    # test_corpus = get_test_corpus()

    # spelling_corpora = grab_spelling_files()

    # spelling_bigrams = calc_all_corpora_bigram(spelling_corpora)

    # confusion_set = parse_confusion_set()

    # spell_check("atheism")

    arg_options = ["sentence", "perplexity", "spell-check", "classification"]

    secondary_opts = {
    "spell-check" : 1,
    "sentence": 2,
    "perplexity" : 2,
    "classification" : 0,
    "test":  (-1)
    }

    #sentence takes [unigram | bigram] and [corpus]
    #perplexity takes [unigram | bigram] and [corpus]
    #spell-check takes [corpus]

    args = sys.argv
    option = args[1]
    if option in arg_options:
        num_secondary = secondary_opts[option]
        if num_secondary == 2:
            ngram = args[2]
            corpus = args[3]
        elif num_secondary == 1:
            corpus = args[2]
        elif num_secondary == 0:
            #classification needs no args
            pass
        else:
            #testing purposes, add any variables needed here
            pass

        if option == "sentence":
            handle_sentence_generation(ngram, corpus)
        elif option == "perplexity":
            handle_perplexity_calculation(ngram, corpus)
        elif option == "spell-check":
            if corpus == "all":
                corpora = grab_files()
                for corpus in corpora:
                    handle_spell_checker(corpus)
            else:
                handle_spell_checker(corpus)
        elif option == "classification":
            topic_classification()
        else:
            # corpora = grab_files()
            # #calc_gt_all_corpora_unigram(corpora)
            # gt_probs = calc_gt_all_corpora_bigram(corpora)
            # reg_probs = calc_all_corpora_bigram(corpora)
            # print(gt_probs["atheism"])
            # print reg_probs
            # json.dump(gt_probs["atheism"], open("gt.json", "w"))
            # json.dump(reg_probs["atheism"], open("reg.json", "w"))
            # calc_gt_all_corpora_bigram(corpora)
            #calc_gt_all_corpora_bigram(corpora)
            #for key in corpora:
            #    count_trigram_tokens(corpora[key])
            # topic_classification()
            pass

    else:
        print("ERROR: Unknown action option")
        sys.exit(1)


