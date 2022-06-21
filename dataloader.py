import pandas as pd


TRAIN_ARGS = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'sliding_window': True,
    'max_seq_length': 64,
    'num_train_epochs': 10, #8
    'train_batch_size': 32,
    'fp16': True,
    'output_dir': '/outputs/',
    'best_model_dir': '/outputs/best_model/',
    'evaluate_during_training': True,
}


def file_to_df(filename):
    df = pd.read_csv(filename,
                    sep = '\t', header = None, keep_default_na = False,
                    names = ['words', 'pos', 'chunk', 'labels'],
                    quoting = 3, skip_blank_lines = False)

    df = df[~df['words'].astype(str).str.startswith('-DOCSTART- ')]
    df['sentence_id'] = (df.words == '').cumsum()

    return df[df.words != '']


def load_files():
    train_df = file_to_df('data/train.txt')
    test_df = file_to_df('data/test.txt')
    dev_df = file_to_df('data/dev.txt')

#    train_df.words = train_df.words.str.lower()
#    test_df.words = test_df.words.str.lower()
#    dev_df.words = dev_df.words.str.lower()

    return train_df, test_df, dev_df

