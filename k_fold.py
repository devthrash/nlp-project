import os
import pickle

from sklearn.model_selection import KFold, train_test_split
from torch import save

from ironydetection import *
from ironydetection.net.encoder import Encoder
from ironydetection.preprocessor import load_csv
from tqdm import tqdm


def run_k_fold(epochs=30, k=5):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    with open(os.path.join('output', 'custom_encoder.pickle'), 'rb') as handle:
        encoder: Encoder = pickle.load(handle)

    dataset_train_path = os.path.join('input', 'datasets', 'train', 'SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt')

    df = load_csv(dataset_train_path)

    all_results = []
    best_loss = float('inf')

    # for train_df, validate_df in (train_test_split(df, test_size=0.2, shuffle=False, random_state=0),):
    for train_idx, validate_idx in tqdm(k_fold.split(df)):
        model = TweetBinaryClassifier(vocab_size=encoder.vocab_size(), embedding_dim=100, hidden_dim=50).to(DEVICE)

        train_dl = dataloader(TweetsDataset('', encoder, df=df.iloc[train_idx]))
        # train_dl = dataloader(TweetsDataset('', encoder, df=train_df))

        validate_dl = dataloader(TweetsDataset('', encoder, df=df.iloc[validate_idx]))
        # validate_dl = dataloader(TweetsDataset('', encoder, df=validate_df))

        all_results.append(train_and_validate(model, train_dl, validate_dl, epochs))
        validation_loss = all_results[-1].copy().sort_values(by='validation_loss')['validation_loss'].values[0]

        if validation_loss < best_loss:
            best_loss = validation_loss

            save(model.state_dict(), os.path.join('output', 'best_model_kfold_cross_validation1.pickle'))

    return all_results


def run_k_fold_pretrained(epochs=30, k=5):
    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    embeddings, encoder = get_pretrained_embeddings_and_create_encoder('glove-twitter-100')

    dataset_train_path = os.path.join('input', 'datasets', 'train', 'SemEval2018-T3-train-taskA_emoji_ironyHashtags.txt')

    df = load_csv(dataset_train_path)

    all_results = []
    best_loss = float('inf')

    for train_idx, validate_idx in tqdm(k_fold.split(df)):
        model = TweetBinaryClassifier(weights=embeddings, embedding_dim=100, hidden_dim=50).to(DEVICE)

        train_dl = dataloader(TweetsDataset('', encoder, df=df.iloc[train_idx]))
        validate_dl = dataloader(TweetsDataset('', encoder, df=df.iloc[validate_idx]))

        all_results.append(train_and_validate(model, train_dl, validate_dl, epochs))
        validation_loss = all_results[-1].copy().sort_values(by='validation_loss')['validation_loss'].values[0]

        if validation_loss < best_loss:
            best_loss = validation_loss

            save(model.state_dict(), os.path.join('output', 'best_model_kfold_cross_validation_pretrained1.pickle'))

    return all_results
