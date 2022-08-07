import pickle
import spacy

nlp = spacy.load('en_core_web_sm')

# dictionary look up
with open('dict_lookup/dictionary_lookup.pkl', 'rb') as f:
    dict_lookup = pickle.load(f)


def tokenize_words(article):
    """ Takes an input and tokenizes it based on the spacy model.

    :param article: Input as an article (i.e. PubMed)
    :return: Returns tokens of the article.
    """
    doc = nlp(article)
    words = [word.text for word in doc]
    
    return words


def dict_match(article_list, dis_chems_list=dict_lookup):
    """ Looks for diseases and chemicals consisting of multiple tokens and unifies it as one based on a dict lookup approach.

    :param article_list: List of articles which takes an input.
    :param dis_chems_list: List of chemicals and diseases.
    :return: Returns list of tokens of the articles with respect to diseases and chemicals with multiple possible tokens.
    """
    article_list = [tokenize_words(i) for i in article_list][0]
    data = ' '.join(article_list)

    for dis_chems in dis_chems_list:
        d = dis_chems.replace(' ', '_')
        data = data.replace(dis_chems, d)

    tokens = data.split()

    for i,t in enumerate(tokens):
        t = t.replace('_', ' ')
        tokens[i] = t

    return tokens