# medical-graph-NER

Biomedial NER Model based on the [BC5CDR (BioCreative V CDR corpus) dataset](https://paperswithcode.com/dataset/bc5cdr)).
BC5CDR corpus consists of 1500 PubMed articles with 4409 annotated chemicals, 5818 diseases.
Diseases are labelled as I-Entity and Chemicals as O-Entity.

We achieved more than 90% measured with the F1-score. 

```
pip install -r requirements.txt
```

Please use a GPU to to train the model. 

### Using the NER-Model

```python
import medGraphNER

mgner = medGraphNER.MedGraphNER() # this can take a lot of time if the model was not trained before
mgner.get_bio_NER("this article is about diabetes")

>>> {'diabetes' : 'I-Entity'} 

```
