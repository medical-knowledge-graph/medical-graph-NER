from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import torch
import os
import logging
import warnings


class BioModel:
    def __init__(self, labels, train_args, import_=True):
        logging.basicConfig(level=logging.DEBUG)
        transformers_logger = logging.getLogger('transformers')
        transformers_logger.setLevel(logging.WARNING)

        if not self.check_gpu():
            self.model = NERModel('bert', 'dmis-lab/biobert-v1.1', labels=labels, use_cuda=False, args=train_args)
        elif import_==True:
            self.model = NERModel('bert', 'dmis-lab/biobert-v1.1', labels=labels, args=train_args)

    def train(self, train_df, test_df, dev_df):
        self.check_gpu()
        self.model.train_model(train_df, eval_data=dev_df)
        result, _, _ = self.model.eval_model(test_df)
        self.save_model(self.model)
        print(result)

    @staticmethod
    def check_gpu():
        if not torch.cuda.is_available():
            warnings.warn("You are not training on a GPU. This may take a lot of time.")
            return False

        return True

    @staticmethod
    def save_model(model):
        dir_name = 'model'
        try:
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
            torch.save(model, "model/model")

        except FileExistsError:
            print("Directory ", dir_name, " already exists or permission denied")

    def load_model(self):
        dir_name = 'model'
        try:
            self.model = torch.load('model/model')
            print("model was loaded")

        except FileExistsError:
            print("Directory ", dir_name, " does not exist or permission denied")

    def predict(self, text):
        pred = self.model.predict([text])

        if pred:
            results = [i for i in pred[0][0] if list(i.values())[0] == ('I-Entity' or 'B-Entity')]
            return results

        return []

print(torch.cuda.is_available())
