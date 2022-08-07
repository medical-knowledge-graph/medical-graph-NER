import pandas as pd

# Parameters for the trainprocess.
TRAIN_ARGS = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 128,
    'num_train_epochs': 12, #8
    'train_batch_size': 32,
    'fp16': True,
    'output_dir': '/outputs/',
    'best_model_dir': '/outputs/best_model/',
    'evaluate_during_training': True,
}


def file_to_df(filename):
    """ Reads a file and transforms it to a readable format and filters by empty entries.

    :param filename: Filename.
    :return: Returns correct format with filtered entries.
    """
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels', 'sentence_id'],
                    quoting = 3, skip_blank_lines = False)

    return df[df.words != '']


def load_files():
    """ Loads train, test and evaluation dataset.

    :return: Returns train, test and evaluation dataset.
    """
    train_df = file_to_df('data/train_new.txt')
    test_df = file_to_df('data/test_new.txt')
    dev_df = file_to_df('data/eval_new.txt')

    return train_df, test_df, dev_df


train_df, test_df, dev_df = load_files()