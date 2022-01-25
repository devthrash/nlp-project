import os
import pickle

from ironydetection import *
from os import path
import torch

from ironydetection.net.encoder import Encoder
from ironydetection.preprocessor.preprocessor import load_csv

torch.manual_seed(0)


def run_test_model():
    with open(os.path.join('output', 'custom_encoder.pickle'), 'rb') as handle:
        encoder: Encoder = pickle.load(handle)

    test_dataset_path = path.join('input', 'datasets', 'goldtest_TaskA', 'SemEval2018-T3_gold_test_taskA_emoji.txt')
    dataloader_test = dataloader(TweetsDataset(test_dataset_path, encoder))

    model = TweetBinaryClassifier(vocab_size=encoder.vocab_size(), embedding_dim=100, hidden_dim=50).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join('output', 'best_model_kfold_cross_validation1.pickle')))
    model.eval()

    test(model, dataloader_test)


def run_test_model_pretrained_embeddings():
    embeddings, encoder = get_pretrained_embeddings_and_create_encoder('glove-twitter-100')

    test_dataset_path = path.join('input', 'datasets', 'goldtest_TaskA', 'SemEval2018-T3_gold_test_taskA_emoji.txt')
    dataloader_test = dataloader(TweetsDataset(test_dataset_path, encoder))

    model = TweetBinaryClassifier(weights=embeddings, hidden_dim=50).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join('output', 'best_model_kfold_cross_validation_pretrained1.pickle')))
    model.eval()

    test(model, dataloader_test)


def test_stats():
    test_dataset_path = path.join('input', 'datasets', 'goldtest_TaskA', 'SemEval2018-T3_gold_test_taskA_emoji.txt')
    df = load_csv(test_dataset_path)

    non_ironic_count = len(df.loc[df['label'] == 0])
    ironic_count = len(df.loc[df['label'] == 1])

    print('Test data')
    print('Non-ironic tweets: {}\tIronic tweets: {}'.format(non_ironic_count, ironic_count))

    print('Random chance: {}'.format((non_ironic_count / len(df))**2 + (ironic_count / len(df))**2))


if __name__ == '__main__':
    test_stats()
    run_test_model()
    run_test_model_pretrained_embeddings()
