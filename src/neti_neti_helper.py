# Machine Learning based approach to find scientific names
# Collection of functions used by name finding algorithms

"""
neti_neti_helper.py

Created by Lakshmi Manohar Akella.
Copyright (c) 2010, 2011, Marine Biological Laboratory.
All rights resersved.

"""
import re

#r_strip = re.compile(u'^(.*?)([\\W\\d_]*)$', re.U | re.M)
#l_strip = re.compile(u'^([\\W\\d_]*)(.*)$', re.U | re.M)

def left_strip(token):
    """Takes a token and strips non alpha characters off the left. It
    returns the stripped string and the number of characters it stripped.

    Arguments:
    token -- a one word token which might have trailing non alpha characters

    """
    i = 0
    while(i < len(token)):
        if(not token[i].isalpha()):
            i = i + 1
        else:
            break
    if(i == len(token)):
        return('', 0)
    return(token[i:], i)

def right_strip(token):
    """This takes a token and strips non alpha characters off the right. It
    returns the stripped string and the number of characters it stripped.

    amount of stripped characters allows to calculate locations of found
    scientific names within the original document

    Arguments:
    t -- a one word token

    """
    i = -1
    while(i >= -len(token)):
        if(not token[i].isalpha()):
            i = i - 1
        else:
            break
    if(i == -1):
        return(token, 0)
    elif(i == -(len(token) + 1)):
        return('', 0)
    else:
        return(token[:i + 1], i + 1)

def strip_token(token):
    """This combines left_strip and right_strip into one method.
    Returns back token without trailing non alpha characters

    Arguments:
    token -- a one word token which might contain trailing non alpha characters
         like parentheses, comma, etc...

    """
    return(right_strip(left_strip(token)[0])[0])

def get_words_slice(words,
        word_index,
        first_char_index,
        second_char_index = None):
    """Returns a requested slice of a word, if possible, 'Null' otherwise

    Arguments:
    words -- list of words
    word_index -- index of a word to slice
    first_char_index -- start of the slice
    second_char_index -- end of the slice
    """
    result = 'Null'
    try:
        if (second_char_index == None):
            result = words[word_index][first_char_index]
        else:
            result = words[word_index][first_char_index:second_char_index]
    except IndexError:
        pass
    if result == []:
        result = 'Null'
    return result

def clean_token(first_word, second_word, third_word):
    """Cleans the tokens.
    Intelligent strip of trigram parts. We are assuming a and b are stripped

    Arguments:
    first_word -- first element of a trigram
    second_word -- second element of a trigram
    third_word -- third element of a trigram

    """
    res = [first_word, second_word, strip_token(third_word)]
    if len(first_word) > 1:
        if first_word[-1] == ".":
            res[0] = left_strip(first_word)[0]
        else:
            res[0] = strip_token(first_word)
    if len(second_word) > 1:
        if second_word[0] + second_word[-1] == "()":
            pass
        elif second_word[-1] == "-":
            res[1] = left_strip(second_word)[0]
        else:
            res[1] = strip_token(second_word)
    return res

def create_index(tokens):
    """Returns a dictionary of indexes for all tockens. Key is a token index in
    the document, Value is the length of the token.

    Arguments:
    token -- list of all tokens from the document checked for scientific names

    """
    length = len(tokens[0]) + 1
    indices_dict = {0:0}
    for i in range(1, len(tokens)):
        indices_dict[i] = length
        length += len(tokens[i]) + 1
    return indices_dict

def remove_trailing_period(string):
    """Returns the string with removed trailing period

    removes period from words that are not an obvious genus abbreviaion
    we assume here that genus is almost always abbreviated to 1 letter

    Arguments:
    string -- a string, usually the first element of a trigram

    """
    #TODO: sometimes people abbreviate genus to 2 or 3 letters,
    # would it be a problem?

    if len(string) > 2:
        return string.rstrip(".")
    else:
        return string

def has_uninomial_ending(word):
    """Returns boolean
    Checks the ending of a word. It returns True if the word has an ending
    characteristic for a uninomial scientific name

    Arguments:
    word -- a word to check

    """
    endings = ['aceae', 'ales', 'eae', 'idae', 'ina', 'inae', 'ineae',
                'ini', 'mycetes', 'mycota', 'mycotina', 'oidea', 'oideae',
                'opsida', 'phyceae', 'phycidae', 'phyta', 'phytin']
    word = remove_trailing_period(word.lower())
    for ending in endings:
        if word.endswith(ending):
            return True
    return False

def resolve_abbreviated_names(names_list, names_set):
    """
    list of names is augmented with versions of names where genus part
    is abbreviated

    """
    name_regex = re.compile('^(.{1})([^ ]+)(.*)$')
    # (B)(etula) (verucosa verucosa)
    names_dict = {}
    resolved_names = []
    abbr_names = [name for name in names_list if (name[1] == "."
        and name[2] == " ")]
    full_names = names_set.difference(set(abbr_names))
    full_names_list = list(full_names)
    for name in full_names_list:
        result = name_regex.match(name).groups()
        names_dict[result[0] + "." + result[2]] = result[1]
    print (names_dict.keys())
    for name in abbr_names:
        if names_dict.has_key(name):
            resolved_names.append(name[0] \
                    + "[" + names_dict[name] + "]" + " " + name[3:])
        else:
            resolved_names.append(name)
    resolved_names += full_names_list
    resolved_names.sort()
    return resolved_names

