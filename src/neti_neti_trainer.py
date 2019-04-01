# Machine Learning based approach to find scientific names
# NetiNetiTrainer is a class which trains a classifier (Naive Bayesian by
# default) to recognize scientific names

"""
neti_neti_trainer.py

Created by Lakshmi Manohar Akella.
Copyright (c) 2010, 2011, Marine Biological Laboratory.
All rights resersved.

"""

import nltk
import random
import os
#from neti_neti_helper import strip_token, get_words_slice
from src.neti_neti_helper import strip_token, get_words_slice

class NetiNetiTrainer:
    """A class that defines the training algorithm and the training files
    and actually trains a natrual language toolkit object (nltk)
    """
    def __init__(self, positive_training_file = "data/names_in_context.txt",
                 negative_training_file = "data/no_names.txt",
                 sci_names_file = "data/white_list.txt",
                 learning_algorithm="NB",
                 sci_names_training_num = 10000,
                 negative_trigrams_num = 5000, context_span=1):
        """Builds and trains the NetiNeti model

        Keyword arguments:
        positive_training_file -- text with scientific names and a text that
            contains them (default "data/names_in_context.txt")
        negative_training_file -- text without scientific names
            (default "data/no_names.txt")
        sci_names_file -- text containing only scientific names -- one name
            per line (default "data/white_list.txt")
        learning_algorithm -- the algorithm to train with
            (default "Naive Bayesian: NB")
        sci_names_training_num -- number of scientific names from
            sci_names_file to use for training (default 10000)
        negative_trigrams_num -- number of negative trigrams to build for
            training (default 5000)
        context_span -- number of surrounding words from either side to use
            for context training (default 1)

        """
        self._positive_training_file = os.path.dirname(os.path.realpath(__file__)) + "/" + positive_training_file
        self._negative_training_file = os.path.dirname(os.path.realpath(__file__)) + "/" + negative_training_file
        self._sci_names_file = os.path.dirname(os.path.realpath(__file__)) + "/" + sci_names_file
        self._sci_names_training_num = sci_names_training_num
        self._negative_trigrams_num = negative_trigrams_num
        self._context_span = context_span
        self.learning_algorithm = learning_algorithm
        self._model = None # model is added during the training

        self._sci_names, self._white_list = self._tokenize_sci_names()
        featuresets = self._get_training_data()
        self._train_classifier_model(featuresets)

    def _tokenize_sci_names(self):
        """Returns a list of random scientific names and a dictionary of
        one-word tokens generated from the scientific names.

        Scientific names are supplied in an external file which has
        several million names, one name per line.

        The collection of tokens derived from those names is stored as a
        dictionary to ensure uniqueness of all tokens and to ensure
        fast access to them. Tokens stored as keys of the dictionary,
        values are irrelevant and are set to 1.
        """
        lines = open(self._sci_names_file,encoding="utf8").readlines()
        all_sci_names = [line.strip() for line in lines]
        random.shuffle(all_sci_names)
        sci_names = all_sci_names[:self._sci_names_training_num]
        white_list = {}
        for sci_name in all_sci_names:
            #TODO after refactoring: change to split() to avoid empty token.
            # Then figure out how to eliminate empty token from other places
            words = sci_name.split(" ")
            for word in words:
                white_list[word.lower()] = 1
        return sci_names, white_list

    def _get_training_data(self):
        """Builds and returns positive and negative feature sets
        for the algorithm

        """

        positive_training_data = self._get_positive_training_data()
        featuresets = self._get_positive_featuresets(positive_training_data)
        featuresets += self._get_negative_featuresets()
        return featuresets

    def _get_negative_featuresets(self):
        """Returns array
        Builds and returns negative feature sets

        """
        featuresets = []
        ndata = open(self._negative_training_file, encoding='utf8').read()
        ntokens = nltk.word_tokenize(ndata)
        neg_trigrams = nltk.trigrams(ntokens)
        index = -1
        trigram_count = 0
        for first_word, second_word, third_word in neg_trigrams:
            if(trigram_count > self._negative_trigrams_num):
                break
            index += 1
            trigram = first_word + " " + second_word + " " + third_word
            bigram = first_word + " " + second_word
            if (first_word[0].isupper() and first_word[1:].islower()
                and second_word.islower()):

                featuresets.append((self.taxon_features(bigram, ntokens,
                    index, 1), 'not-a-taxon'))
                featuresets.append((self.taxon_features(first_word, ntokens,
                    index, 0), 'not-a-taxon'))
                # TODO find out if it makes sense to move it as a separate
                # if statement
                if(third_word.islower()):
                    trigram_count += 1
                    featuresets.append((self.taxon_features(trigram, ntokens,
                                        index, 2), 'not-a-taxon'))
            #TODO after refactoring: it might make sense to generate unigram
            # once, looks like we have duplicated unigrams here
            if(second_word[0].isupper() and second_word[1:].islower()):
                featuresets.append((self.taxon_features(second_word, ntokens,
                    index + 1, 0), 'not-a-taxon'))
            if(third_word[0].isupper() and third_word[1:].islower()):
                featuresets.append((self.taxon_features(third_word, ntokens,
                    index + 2, 0), 'not-a-taxon'))
        random.shuffle(featuresets)
        return(featuresets)

    def _get_positive_training_data(self):
        """Returns list of data for positive training"""
        data = []
        for line in open(self._positive_training_file,encoding="utf8"):
            if line.strip():
                name, context = line.split("---", 1)
            data.append({ 'name' : name.strip(), 'context' : context.strip() })
        for sci_name in self._sci_names:
            if sci_name:
                data.append({ 'name' : sci_name, 'context' : sci_name })
        return data

    def _get_positive_featuresets(self, positive_training_data):
        """ Creates features set from prepared positive training data """
        featuresets = []
        for data in positive_training_data:
            name = data['name']
            context = data['context']
            context_tokens = nltk.word_tokenize(context)
            name_tokens = nltk.word_tokenize(name)
            try:
                index = context_tokens.index(name_tokens[0])
            except ValueError:
                index = 0
                #TODO after refactoring: should index be 0 or
                # it should be -1 or something?
            span = len(name_tokens) - 1

            features = self.taxon_features(name, context_tokens, index, span)
            featuresets.append((features, 'taxon'))

            # create abbreviated version of the name (e.g. Aus bus -> A. bus)
            #and make features.
            if(len(name_tokens[0]) > 1 and name_tokens[0][1] != "."):
                abbr_name = name_tokens[0][0]+". " + " ".join(name_tokens[1:])
                features = self.taxon_features(abbr_name,
                    context_tokens, index,span)
                featuresets.append((features, 'taxon'))
        return featuresets

    def taxon_features(self, token, context_array, index, span):
        """Returns a dictionary of features

        Arguments:
        token -- a name string consisting of 1-3 words
        context -- list of words surrounding the token
        index -- index where the token happens in the document
        span -- span + index indicates the position of the last word
        """
        token = token.strip()
        words = token.split(" ")
        context_span = self._context_span
        features = {}
        vowels = ['a', 'e', 'i', 'o', 'u']
        last_chars = ['a', 'i', 's', 'm'] #last letter (LL) weight
        last_chars_reduced = ['e', 'o']   # Reduced LL weight
        last_chars_all = last_chars + last_chars_reduced
        penultimate_chars = ['i', 'u']    # penultimate L weight
        string_weight = 0
        weight_increment = 5

        features['last3_first_word']  = get_words_slice(words, 0, -3, 1000)
        features['last3_second_word'] = get_words_slice(words, 1, -3, 1000)
        features['last3_third_word']  = get_words_slice(words, 2, -3, 1000)
        features['last2_first_word']  = get_words_slice(words, 0, -2, 1000)
        features['last2_second_word'] = get_words_slice(words, 1, -2, 1000)
        features['last2_third_word']  = get_words_slice(words, 2, -2, 1000)
        features['char1_first_word']  = get_words_slice(words, 0, 0)
        features['char-1_first_word'] = get_words_slice(words, 0, -1)
        features['char2_first_word']  = get_words_slice(words, 0, 1)
        features['char-2_first_word'] = get_words_slice(words, 0, -2)

        features["char-1_first_word_in_lc"] = (features['char-1_first_word']
                in last_chars)
        if features["char-1_first_word_in_lc"]:
            string_weight += weight_increment

        features["char-1_first_word_in_lcr"] = (features['char-1_first_word']
                in last_chars_reduced)
        if features["char-1_first_word_in_lcr"]:
            string_weight += weight_increment - 3

        char_last_second = get_words_slice(words, 1, -1)

        features["char-1_second_word_in_lc"] = char_last_second in last_chars
        if features["char-1_second_word_in_lc"]:
            string_weight += weight_increment

        features["char-1_second_word_in_lcr"] = (char_last_second
                in last_chars_reduced)
        if features["char-1_second_word_in_lcr"]:
            string_weight += weight_increment - 3

        features["char-1_third_word_in_lca"] = (get_words_slice(words, 2, -1)
                in last_chars_all)
        if features["char-1_third_word_in_lca"]:
            string_weight += weight_increment - 2

        features["char-2_third_word_in_lca"] = (get_words_slice(words, 2, -2)
                in last_chars_all)

        features["char-1_first_word_in_vwl"] = (features['char-1_first_word']
                in vowels)
        features["char-2_first_word_in_lc"]  = (features['char-2_first_word']
                in last_chars)
        features["char-2_first_word_in_lcr"] = (features['char-2_first_word']
                in last_chars_reduced)
        features["char-2_first_word_in_pc"]  = (features['char-2_first_word']
                in penultimate_chars)
        char_before_last_second = get_words_slice(words, 1, -2)

        # TODO BUG!!!! remove after refactoring
        # features["char-2_first_word_in_lc"]  = char_before_last_second
        # end BUG

        features["char-2_second_word_in_lc"]  = (char_before_last_second
                in last_chars)
        features["char-2_second_word_in_lcr"] = (char_before_last_second
                in last_chars_reduced)
        features["char-2_second_word_in_pc"]  = (char_before_last_second
                in penultimate_chars)

        # TODO BUG!!!! remove after refactoring
        # features["char-2_first_word_in_pc"] = char_before_last_second
        # end BUG

        #features["first_word_in_wl"]  = \
            #self._white_list.has_key(words[0].lower())
        if words[0].lower() not in self._white_list:
            features["first_word_in_wl"]=None

        #features["second_word_in_wl"] = \
          #self._white_list.has_key(get_words_slice(words, 1, 0, 1000).lower())

        if get_words_slice(words, 1, 0, 1000).lower() not in self._white_list:
            features["second_word_in_wl"]=None

        #features["third_word_in_wl"]  = \
          #self._white_list.has_key(get_words_slice(words, 2, 0, 1000).lower())

        if get_words_slice(words, 2, 0, 1000).lower() not in self._white_list:
            features["third_word_in_wl"] = None


        for i in range(context_span):
            features[str(i + 1) + "_context"] = \
                strip_token(get_words_slice(context_array,
                    index + span + i + 1, 0, 1000))
            if(index + i - context_span < 0):
                features[str(i - context_span) + "_context"] = 'Null'
            else:
                features[str(i - context_span) + "_context"] = \
                    strip_token(get_words_slice(context_array,
                        index + i - context_span, 0, 1000))
        try:
            features["1up_2dot_rest_lower_case"] = (token[0].isupper() and
                                            token[1] is "." and
                                            token[2] is " " and
                                            token[3:].islower())
        except IndexError:
            features["1up_2dot_rest_lower_case"] = False
        features["token"] = token
        for vowel in'aeiou':
            features["count(%s)" % vowel] = token.lower().count(vowel)
            features["has(%s)" % vowel] = vowel in token
        features["capitalized_first_word"] = \
            token[0].isupper() and token[1:].islower()

        if(string_weight > 18):
            features["string_weight"] = 'A'
        elif(string_weight > 14):
            features["string_weight"] = 'B'
        elif(string_weight > 9):
            features["string_weight"] = 'C'
        elif(string_weight > 4):
            features["string_weight"] = 'D'
        else:
            features["string_weight"] = 'F'
        return features

    def _train_classifier_model(self, featuresets):
        """This changes the algorithm that nltk uses to train the model.

        Arguments:
        featuresets -- array of features generated for training

        """
        model = None
        if(self.learning_algorithm == "NB"):
            model = nltk.NaiveBayesClassifier.train(featuresets)
        elif(self.learning_algorithm == "MaxEnt"):
            model = nltk.MaxentClassifier.train(featuresets, "MEGAM",
                                                 max_iter=15)
        elif(self.learning_algorithm == "DecisionTree"):
            model = nltk.DecisionTreeClassifier.train(featuresets, 0.05)
        self._model = model

    def get_model(self):
        """An accessor method for the model."""
        return self._model

