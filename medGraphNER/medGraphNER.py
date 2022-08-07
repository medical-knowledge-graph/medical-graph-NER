import dataloader
import bioModel
import NERtokens
import os.path

class MedGraphNER:
    """ Class to initiate loading and training of the model.
    """

    def __init__(self):
        """ Create or load the model based on the collected BC5CDR Data.
        """
        train_args = dataloader.TRAIN_ARGS
        train_df = dataloader.train_df
        dev_df = dataloader.dev_df
        test_df = dataloader.test_df
        labels = list(train_df['labels'].unique())

        if os.path.isfile('model/model'):
            print("loading model....")
            self.model = bioModel.BioModel(labels, train_args)
        else:
            print("model was not found. Starting training...")
            self.model = bioModel.BioModel(labels, train_args)
            self.model.train(train_df, test_df, dev_df)

    def get_bio_NER(self, text):
        """ Takes an input to create tokens and pass it to the predict function.

        :param text:
        :return:
        """
        text = NERtokens.dict_match([text])
        return self.model.predict(text)


if __name__ == "__main__":
    mgner = MedGraphNER()
    article = "Tricuspid valve regurgitation and lithium carbonate toxicity in a newborn infant."
    mgner.get_bio_NER(article)
