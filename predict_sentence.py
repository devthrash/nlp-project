import os
import pickle

import torch

from ironydetection import *
from ironydetection.net.encoder import Encoder


def main(sentence):
    with open(os.path.join('output', 'custom_encoder.pickle'), 'rb') as handle:
        encoder: Encoder = pickle.load(handle)

    model = TweetBinaryClassifier(vocab_size=encoder.vocab_size(), embedding_dim=100, hidden_dim=50).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join('output', 'best_model_kfold_cross_validation1.pickle')))
    model.eval()
    # embeddings, encoder = get_pretrained_embeddings_and_create_encoder('glove-twitter-100')
    #
    # model = TweetBinaryClassifier(embeddings, hidden_dim=50).to(DEVICE)
    # model.load_state_dict(torch.load(os.path.join('output', 'best_model_1')))
    # model.eval()

    print(predict_sentence(sentence, model=model, encoder=encoder))


if __name__ == '__main__':
    main('this is an ironic post #lol #irony')
