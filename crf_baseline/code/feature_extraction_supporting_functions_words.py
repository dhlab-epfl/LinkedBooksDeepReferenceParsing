# -*- coding: utf-8 -*-
"""
Extraction of Features from words, used to parse references

Inspired from CRFSuite by Naoaki Okazaki: http://www.chokkan.org/software/crfsuite/.
"""
__author__ = """Naoaki Okazaki, Giovanni Colavizza"""

import string, re

def get_shape(token):
    r = ''
    for c in token:
        if c.isupper():
            r += 'U'
        elif c.islower():
            r += 'L'
        elif c.isdigit():
            r += 'D'
        elif c in ('.', ','):
            r += '.'
        elif c in (';', ':', '?', '!'):
            r += ';'
        elif c in ('+', '-', '*', '/', '=', '|', '_'):
            r += '-'
        elif c in ('(', '{', '[', '<'):
            r += '('
        elif c in (')', '}', ']', '>'):
            r += ')'
        else:
            r += c
    return r

def degenerate(src):
    dst = ''
    for c in src:
        if not dst or dst[-1] != c:
            dst += c
    return dst

def get_type(token):
    T = (
        'AllUpper', 'AllDigit', 'AllSymbol',
        'AllUpperDigit', 'AllUpperSymbol', 'AllDigitSymbol',
        'AllUpperDigitSymbol',
        'InitUpper',
        'AllLetter',
        'AllAlnum',
        )
    R = set(T)
    if not token:
        return 'EMPTY'

    for i in range(len(token)):
        c = token[i]
        if c.isupper():
            R.discard('AllDigit')
            R.discard('AllSymbol')
            R.discard('AllDigitSymbol')
        elif c.isdigit() or c in (',', '.'):
            R.discard('AllUpper')
            R.discard('AllSymbol')
            R.discard('AllUpperSymbol')
            R.discard('AllLetter')
        elif c.islower():
            R.discard('AllUpper')
            R.discard('AllDigit')
            R.discard('AllSymbol')
            R.discard('AllUpperDigit')
            R.discard('AllUpperSymbol')
            R.discard('AllDigitSymbol')
            R.discard('AllUpperDigitSymbol')
        else:
            R.discard('AllUpper')
            R.discard('AllDigit')
            R.discard('AllUpperDigit')
            R.discard('AllLetter')
            R.discard('AllAlnum')

        if i == 0 and not c.isupper():
            R.discard('InitUpper')

    for tag in T:
        if tag in R:
            return tag
    return 'NO'

def get_2d(token):
    return len(token) == 2 and token.isdigit()

def get_4d(token):
    return len(token) == 4 and token.isdigit()

def get_parYear(token):
    if token[0] == '(' and token[-1] == ')':
        if get_4d(token[1:-1]) or get_2d(token[1:-1]):
            return True
    return False

# if both digit and alphabetic
def get_da(token):
    bd = False
    ba = False
    for c in token:
        if c.isdigit():
            bd = True
        elif c.isalpha():
            ba = True
        else:
            return False
    return bd and ba

def get_dand(token, p):
    bd = False
    bdd = False
    for c in token:
        if c.isdigit():
            bd = True
        elif c == p:
            bdd = True
        else:
            return False
    return bd and bdd

def get_all_other(token):
    for c in token:
        if c.isalnum():
            return False
    return True

def get_capperiod(token):
    return len(token) == 2 and token[0].isupper() and token[1] == '.'

def contains_upper(token):
    b = False
    for c in token:
        b |= c.isupper()
    return b

def contains_lower(token):
    b = False
    for c in token:
        b |= c.islower()
    return b

def contains_alpha(token):
    b = False
    for c in token:
        b |= c.isalpha()
    return b

def contains_digit(token):
    b = False
    for c in token:
        b |= c.isdigit()
    return b

def contains_symbol(token):
    b = False
    for c in token:
        b |= ~c.isalnum()
    return b

# abbreviations
def is_abbr(token):
    b = False
    if "." in token:
        for p in string.punctuation:
            token = token.replace(p, '')
        if len(token) < 2 & len(token) > 0:
            b = True
        elif len(token) == 2:
            if token[0] == token[1]:
                b = True
    return b

# alternative
# average frequency of abbreviations
# pattern from http://stackoverflow.com/questions/17779771/finding-acronyms-using-regex-in-python
def abbr_pattern(token):
    b = False

    if token is None or len(token) < 1:
        return b

    pattern = r'(?:(?<=\.|\s)[A-Z]\.)+'
    counter = re.search(pattern, token)
    if counter:
        return True
    return b

# New
def is_roman(token):
    b = True
    for p in string.punctuation:
        token = token.replace(p, '')
    for c in token:
        b &= c.lower() in ['i', 'x', 'v', 'c', 'l', 'm', 'd']
    return b

# Return true if a sequence of at least 2 characters matches with roman numbers
def contains_roman(token):
    for n,c in enumerate(token[1:]):
        if c.isupper() and c.lower() in ['i', 'x', 'v', 'c', 'l', 'm', 'd']:
            if token[n-1].isupper() and token[n-1].lower() in ['i', 'x', 'v', 'c', 'l', 'm', 'd']:
                return True
    return False

# is interval, e.g. 1900-10
def is_interval(token):
    b = True
    if "-" in token:
        for x in token.split("-"):
            try:
                int(x)
            except:
                b = False
    return b

# measure the punctuation frequency of a piece of text
def punctuation(text, norm=True):

    if text is None or len(text) < 1:
        return 0

    counter = 0

    for w in range(len(text)):
        if text[w] in string.punctuation:
            counter += 1

    return counter/len(text) if norm else counter

# measure the number frequency of a piece of text
def numbers(text, norm=True):

    if text is None or len(text) < 1:
        return 0

    counter = 0
    for w in text.split():
        for p in string.punctuation:
            w = w.replace(p,"")
        try:
            int(w)
        except:
            continue
        counter += 1

    return counter/len(text) if norm else counter

# frequency of upper case letters
def upper_case(text, norm=True):

    if text is None or len(text) < 1:
        return 0

    counter = 0
    for n in range(len(text)):
        if text[n].isupper():
            counter += 1

    return counter/len(text) if norm else counter

# frequency of lower case letters
def lower_case(text, norm=True):

    if text is None or len(text) < 1:
        return 0

    counter = 0
    for n in range(len(text)):
        if text[n].islower():
            counter += 1

    return counter/len(text) if norm else counter

# number of chars (with whitespace)
def chars(text):

    if text is None or len(text) < 1:
        return 0

    return len(text)

# is abbreviation? (any word of len <= 3 with a dot at the end)
def abbr(text):

    if text is None or len(text) < 1:
        return 0

    if len(text) <= 4 and text[-1] == ".":
        return True
    else:
        return False

# Boolean generation functions based on generic input
def b(v):
    return 'yes' if v else 'no'