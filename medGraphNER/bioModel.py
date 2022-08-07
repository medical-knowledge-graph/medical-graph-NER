from simpletransformers.ner import NERModel
from transformers import AutoTokenizer
import torch
import os
import logging
import warnings


class BioModel:
    """ Includes all functionalities of the ML-model.
    """

    def __init__(self, labels, train_args):
        """ Checks if a created model exists or loads it based on BioBert v1.1.

        :param labels: Labels of the traindataset
        :param train_args: Trainparameters for the train process.
        """
        logging.basicConfig(level=logging.DEBUG)
        transformers_logger = logging.getLogger('transformers')
        transformers_logger.setLevel(logging.WARNING)

        if not self.check_gpu() and not os.path.isfile("model/model"):
            self.model = NERModel('bert', 'dmis-lab/biobert-v1.1', labels=labels, use_cuda=False, args=train_args)
        elif not os.path.isfile("model/model"):
            self.model = NERModel('bert', 'dmis-lab/biobert-v1.1', labels=labels, args=train_args)
        else:
            self.load_model()

    def train(self, train_df, test_df, dev_df):
        """ Finetunes model based on BioBert v1.1.

        :param train_df: Training dataset
        :param test_df: Test dataset
        :param dev_df: Development dataset
        """
        self.check_gpu()
        self.model.train_model(train_df, eval_data=dev_df)
        result, _, _ = self.model.eval_model(test_df)
        self.save_model(self.model)
        print(result)

    @staticmethod
    def check_gpu():
        """ Checks if GPU exists and if not, it warns the user about timecomplexity.

        :return: Returns True if GPU exists, otherwise returns False.
        """
        if not torch.cuda.is_available():
            warnings.warn("You are not training on a GPU. This may take a lot of time.")
            return False

        return True

    @staticmethod
    def save_model(model):
        """ Saves model as soon as the training is finished.

        :param model: Trained model.
        """
        dir_name = 'model'
        try:
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
            torch.save(model, "model/model")

        except FileExistsError:
            print("Directory ", dir_name, " already exists or permission denied")

    def load_model(self):
        """ Loads trained model.
        """
        dir_name = 'model'

        try:
            if not torch.cuda.is_available():
                warnings.warn("You are not predicting on a GPU.")
                self.model = torch.load('model/model', map_location = 'cpu')
            else:
                self.model = torch.load('model/model')
            print("model was loaded")

        except FileExistsError:
            print("Directory ", dir_name, " does not exist or permission denied")

    def predict(self, text):
        """ Predicts Diseases and Chemicals into right format based on the input.

        :param text: Input of the user.
        :return: Returns predicted chemicals and diseases.
        """
        pred = self.model.predict([text], split_on_space=False)

        if pred:
            results = [i for i in pred[0][0] if list(i.values())[0] == 'Disease' or list(i.values())[0] == 'Chemical']
            return results

        return []


print(torch.cuda.is_available())