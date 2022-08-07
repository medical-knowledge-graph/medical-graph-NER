import pickle
import spacy

nlp = spacy.load('en_core_web_sm')

with open('dict_lookup/dictionary_lookup.pkl', 'rb') as f:
    dict_lookup = pickle.load(f)


def tokenize_words(article):
    doc = nlp(article)
    words = [word.text for word in doc]
    
    return words


def dict_match(article_list, disease_list=dict_lookup):
    article_list = [tokenize_words(i) for i in article_list][0]
    data = ' '.join(article_list)

    for disease in disease_list:
        d = disease.replace(' ', '_')
        data = data.replace(disease, d)

    tokens = data.split()

    for i,t in enumerate(tokens):
        t = t.replace('_', ' ')
        tokens[i] = t

    return tokens