# medical-graph-NER

Biomedial NER Model based on the [BC5CDR (BioCreative V CDR corpus) dataset](https://paperswithcode.com/dataset/bc5cdr)).
BC5CDR corpus consists of 1500 PubMed articles with 4409 annotated chemicals, 5818 diseases.
Diseases & Chemicals are labelled as I-Entities and O-Entities.

We achieved more than 90.3% for the F1-score, which outperforms the SciSpacy model 'en_ner_bc5cdr_md'	(F1-scoe: 84.70%). (see 'Evaluation of our model.ipynb')

```
pip install -r requirements.txt
```
Download the medium spacy model:
```
python -m spacy download en_core_web_sm
```

Please setup CUDA based on the Environment and Pytorch version. For instance, conda:
```
conda install pytorch torchvision cudatoolkit=1X.X -c pytorch
```

### Using the NER-Model

```python

from medGraphNER import medGraphNER

mgner = medGraphNER.MedGraphNER()  # this can take a lot of time if the model was not trained before
mgner.get_bio_NER("this article is about diabetes")

>> > {'diabetes': 'Disease'}

```