import pandas as pd


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
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels', 'sentence_id'],
                    quoting = 3, skip_blank_lines = False)

    return df[df.words != '']


def load_files():
    train_df = file_to_df('data/train_new.txt')
    test_df = file_to_df('data/test_new.txt')
    dev_df = file_to_df('data/eval_new.txt')

    return train_df, test_df, dev_df


train_df, test_df, dev_df = load_files()