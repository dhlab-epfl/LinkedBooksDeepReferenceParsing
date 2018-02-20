# -*- coding: utf-8 -*-
"""
Generator of features, relies on feature_extraction_supporting_functions_words

Compatible with sklearn_crfsuite
"""
__author__ = """Giovanni Colavizza"""

from code.feature_extraction_supporting_functions_words import *

def generate_featuresFull(word,n_id,defval=''):
    """
    Creates a set of features for a given token.

    :param word: token
    :param n_id: reference number of token in feature window
    :param def_val: default value for missing features
    :return: The token and its features
    """

    v = {'w%d'%n_id: word}
    # Lowercased token.
    v['wl%d'%n_id] = v['w%d'%n_id].lower()
    # Token shape.
    v['shape%d'%n_id] = get_shape(v['w%d'%n_id])
    # Token shape degenerated.
    v['shaped%d'%n_id] = degenerate(v['shape%d'%n_id])
    # Token type.
    v['type%d'%n_id] = get_type(v['w%d'%n_id])

    # Prefixes (length between one to four).
    v['p1%d'%n_id] = v['w%d'%n_id][0] if len(v['w%d'%n_id]) >= 1 else defval
    v['p2%d'%n_id] = v['w%d'%n_id][:2] if len(v['w%d'%n_id]) >= 2 else defval
    v['p3%d'%n_id] = v['w%d'%n_id][:3] if len(v['w%d'%n_id]) >= 3 else defval
    v['p4%d'%n_id] = v['w%d'%n_id][:4] if len(v['w%d'%n_id]) >= 4 else defval

    # Suffixes (length between one to four).
    v['s1%d'%n_id] = v['w%d'%n_id][-1] if len(v['w%d'%n_id]) >= 1 else defval
    v['s2%d'%n_id] = v['w%d'%n_id][-2:] if len(v['w%d'%n_id]) >= 2 else defval
    v['s3%d'%n_id] = v['w%d'%n_id][-3:] if len(v['w%d'%n_id]) >= 3 else defval
    v['s4%d'%n_id] = v['w%d'%n_id][-4:] if len(v['w%d'%n_id]) >= 4 else defval

    # Two digits
    v['2d%d'%n_id] = b(get_2d(v['w%d'%n_id]))
    # Four digits.
    v['4d%d'%n_id] = b(get_4d(v['w%d'%n_id]))
    # Has a number with parentheses
    v['4d%d'%n_id] = b(get_parYear(v['w%d'%n_id]))
    # Alphanumeric token.
    v['d&a%d'%n_id] = b(get_da(v['w%d'%n_id]))
    # Digits and '-'.
    v['d&-%d'%n_id] = b(get_dand(v['w%d'%n_id], '-'))
    # Digits and '/'.
    v['d&/%d'%n_id] = b(get_dand(v['w%d'%n_id], '/'))
    # Digits and ','.
    v['d&,%d'%n_id] = b(get_dand(v['w%d'%n_id], ','))
    # Digits and '.'.
    v['d&.%d'%n_id] = b(get_dand(v['w%d'%n_id], '.'))
    # A uppercase letter followed by '.'
    v['up%d'%n_id] = b(get_capperiod(v['w%d'%n_id]))

    # An initial uppercase letter.
    v['iu%d'%n_id] = b(v['w%d'%n_id] and v['w%d'%n_id][0].isupper())
    # All uppercase letters.
    v['au%d'%n_id] = b(v['w%d'%n_id].isupper())
    # All lowercase letters.
    v['al%d'%n_id] = b(v['w%d'%n_id].islower())
    # All digit letters.
    v['ad%d'%n_id] = b(v['w%d'%n_id].isdigit())
    # All other (non-alphanumeric) letters.
    v['ao%d'%n_id] = b(get_all_other(v['w%d'%n_id]))

    # Contains a uppercase letter.
    v['cu%d'%n_id] = b(contains_upper(v['w%d'%n_id]))
    # Contains a lowercase letter.
    v['cl%d'%n_id] = b(contains_lower(v['w%d'%n_id]))
    # Contains a alphabet letter.
    v['ca%d'%n_id] = b(contains_alpha(v['w%d'%n_id]))
    # Contains a digit.
    v['cd%d'%n_id] = b(contains_digit(v['w%d'%n_id]))
    # Contains a symbol.
    v['cs%d'%n_id] = b(contains_symbol(v['w%d'%n_id]))

    # Is abbreviation.
    v['ab%d'%n_id] = b(is_abbr(v['w%d'%n_id]))
    # Is abbreviation 2
    v['ab2%d'%n_id] = b(abbr(v['w%d'%n_id]))
    # Is Roman number.
    v['ro%d'%n_id] = b(is_roman(v['w%d'%n_id]))
    v['cont_ro%d'%n_id] = b(contains_roman(v['w%d'%n_id]))
    # Is Interval.
    v['int%d'%n_id] = b(is_interval(v['w%d'%n_id]))

    return v

def generate_featuresLight(word,n_id,defval=''):
    """
    Lightweight version of the above.

    :param word: token
    :param n_id: reference number of token in feature window
    :param def_val: default value for missing features
    :return: The token and its features
    """

    v = {'w%d'%n_id: word}
    # Lowercased token.
    v['wl%d'%n_id] = v['w%d'%n_id].lower()
    # Token shape.
    v['shape%d'%n_id] = get_shape(v['w%d'%n_id])
    # Token shape degenerated.
    v['shaped%d'%n_id] = degenerate(v['shape%d'%n_id])
    # Token type.
    v['type%d'%n_id] = get_type(v['w%d'%n_id])

    # Prefixes (length between one to four).
    v['p1%d'%n_id] = v['w%d'%n_id][0] if len(v['w%d'%n_id]) >= 1 else defval
    v['p2%d'%n_id] = v['w%d'%n_id][:2] if len(v['w%d'%n_id]) >= 2 else defval

    # Suffixes (length between one to four).
    v['s1%d'%n_id] = v['w%d'%n_id][-1] if len(v['w%d'%n_id]) >= 1 else defval
    v['s2%d'%n_id] = v['w%d'%n_id][-2:] if len(v['w%d'%n_id]) >= 2 else defval

    # Two digits
    v['2d%d'%n_id] = b(get_2d(v['w%d'%n_id]))
    # Four digits.
    v['4d%d'%n_id] = b(get_4d(v['w%d'%n_id]))
    # Alphanumeric token.
    v['d&a%d'%n_id] = b(get_da(v['w%d'%n_id]))
    # Digits and '-'.
    v['d&-%d'%n_id] = b(get_dand(v['w%d'%n_id], '-'))
    # Digits and '/'.
    v['d&/%d'%n_id] = b(get_dand(v['w%d'%n_id], '/'))
    # Digits and ','.
    v['d&,%d'%n_id] = b(get_dand(v['w%d'%n_id], ','))
    # Digits and '.'.
    v['d&.%d'%n_id] = b(get_dand(v['w%d'%n_id], '.'))
    # A uppercase letter followed by '.'
    v['up%d'%n_id] = b(get_capperiod(v['w%d'%n_id]))

    # An initial uppercase letter.
    v['iu%d'%n_id] = b(v['w%d'%n_id] and v['w%d'%n_id][0].isupper())
    # All uppercase letters.
    v['au%d'%n_id] = b(v['w%d'%n_id].isupper())
    # All lowercase letters.
    v['al%d'%n_id] = b(v['w%d'%n_id].islower())
    # All digit letters.
    v['ad%d'%n_id] = b(v['w%d'%n_id].isdigit())
    # All other (non-alphanumeric) letters.
    v['ao%d'%n_id] = b(get_all_other(v['w%d'%n_id]))

    # Contains a uppercase letter.
    v['cu%d'%n_id] = b(contains_upper(v['w%d'%n_id]))
    # Contains a lowercase letter.
    v['cl%d'%n_id] = b(contains_lower(v['w%d'%n_id]))
    # Contains a alphabet letter.
    v['ca%d'%n_id] = b(contains_alpha(v['w%d'%n_id]))
    # Contains a digit.
    v['cd%d'%n_id] = b(contains_digit(v['w%d'%n_id]))
    # Contains a symbol.
    v['cs%d'%n_id] = b(contains_symbol(v['w%d'%n_id]))

    # Is abbreviation.
    v['ab%d'%n_id] = b(is_abbr(v['w%d'%n_id]))
    # Is abbreviation 2
    v['ab2%d'%n_id] = b(abbr(v['w%d'%n_id]))
    # Is Roman number.
    v['ro%d'%n_id] = b(is_roman(v['w%d'%n_id]))
    # Is Interval.
    v['int%d'%n_id] = b(is_interval(v['w%d'%n_id]))

    # remove tags and lowercase: language independence
    del v['w%d'%n_id]
    del v['wl%d'%n_id]

    return v



def word2features(sequence, i, extra_labels=[], window=2, feature_function=generate_featuresFull):
    """
    Takes a dataset from a specific document and exports its features for parsing.

    :param sequence: a list of tokens
    :param i: index of token in sequence
    :param extra_labels: list of labels to assign to the text
    :param window: window to consider of preceding and following tokens (e.g. 2 means features for tokens -2 to 2 included will be generated)
    :return: dictionary of token features
    """

    """
    Template of data coming in (4 sequences):

        ['piGNATTi', 'T', '.,', 'Le', 'pitture', 'di', 'Paolo', 'Vero', '-'],
        ['nese', 'nella', 'chiesa', 'di', 'S', '.', 'Sebastiano', 'in'],
        ['Venezia', ',', 'Milano', '1966', '.'],
        ['piGNATTi', 't', '.,', 'Paolo', 'Veronese', ',', 'Milano']]

    """

    if len(extra_labels) > 0:
        assert len(text) == len(extra_labels)

    word = sequence[i]
    position_in_sequence = i


    features = feature_function(word, 0)
    features.update({
        'position': position_in_sequence,
        })

    # Extra Labels to add
    if len(extra_labels) > 0:
        features.update({'tag': extra_labels[i]})


    if i == 0:
        features['BOS'] = True # Begin of Sequence
    else:
        for n in range(-window,0):
            if i+n >= 0:
                word = sequence[i+n]
                features.update(feature_function(word,n))
                # Extra labels
                if len(extra_labels) > 0:
                    features.update({"tag%s"%n:extra_labels[i+n]})

    if i == len(sequence)-1:
        features['EOS'] = True # End of sequence
        for n in range(1,window+1):
            if i+n < len(sequence)-1:
                word = sequence[i+n]
                features.update(feature_function(word,n))
                # Extra labels
                if len(extra_labels) > 0:
                    features.update({"tag%s"%n:extra_labels[i+n]})


    return features
