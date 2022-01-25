import os

import torch
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
import pandas as pd
from ironydetection.preprocessor import DataframePreprocessor

__all__ = [
    'DEVICE', 'TweetBinaryClassifier', 'train_one_epoch', 'test', 'predict_sentence', 'validate', 'train_and_validate'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'


class TweetBinaryClassifier(nn.Module):
    def __init__(self, weights=None, vocab_size=None, embedding_dim=None, hidden_dim=64):
        super().__init__()

        if weights is not None:
            # use gensim word vectors as weights for embedding layer
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        elif vocab_size is not None and embedding_dim is not None:
            # we train it ourselves
            self.embedding = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=vocab_size)
        else:
            raise AttributeError('Weights or vocab_size and embedding_dim should be provided')

        self.lstm = nn.LSTM(input_size=weights.shape[1] if weights is not None else embedding_dim,  # size of word
                                                                                                    # vectors
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True
                            )
        self.drop = nn.Dropout(p=0.5)
        # add only one output because we have only two classes: 0 and 1 so we can use only one output and round it later
        # for comparing with 0 and 1
        # input dimension for fc layer is hidden_dim times 2 because we set bidirectional=True for the LSTM layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.activation = nn.Sigmoid()

    def forward(self, inputs, lengths):
        # inputs shape should be (batch size, number of tokens in sentence)

        embedded = self.embedding(inputs)
        # handle padding values
        # @todo enforce_sorted=True after changing net collate fn to sort inputs by length (may help with the
        # performance)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        lstm_output, (hidden, cell) = self.lstm(packed_embedded)

        # extract the latest two vectors
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        hidden = self.drop(hidden)

        # classify
        dense_outputs = self.fc(hidden)
        outputs = self.activation(dense_outputs)

        return outputs


def train_one_epoch(model: TweetBinaryClassifier, train_loader, disable_pbar=False):
    model.train()
    criterion = nn.BCELoss().to(DEVICE)

    # embeddings layer shouldn't be trained because we use pretrained vectors
    parameters = [param for param in model.parameters() if param.requires_grad]

    # optimizer = optim.SGD(parameters, lr=0.001, momentum=0.90)
    optimizer = optim.Adam(parameters, weight_decay=1e-4)

    train_loss, train_accuracy = 0.0, 0.0

    pbar = tqdm(train_loader, disable=disable_pbar)
    for inputs, lengths, labels in pbar:
        # clear the gradients
        optimizer.zero_grad()

        # forward pass
        # squeeze output so instead of (batch size, 1) we get a flattened array
        outputs = model(inputs.to(DEVICE), lengths).squeeze()

        # get loss
        loss = criterion(outputs.squeeze(), labels.to(DEVICE))

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()

        # save stats
        train_loss += loss.item()

        # round outputs so we have 0 or 1
        predicted_labels = torch.round(outputs)
        # move predictions to CPU se we can convert the tensor to numpy
        train_accuracy_this_batch = accuracy_score(predicted_labels.detach().cpu().numpy(), labels)
        train_accuracy += train_accuracy_this_batch

        pbar.set_description(desc='Loss={:.6f}, acc={:.6f}'.format(loss.item(), train_accuracy_this_batch))

    train_loss /= len(train_loader)
    train_accuracy /= len(train_loader)

    print('Epoch train loss={}\ttrain accuracy={}'.format(train_loss, train_accuracy))

    # return metrics so we can plot them or whatever
    return train_loss, train_accuracy


def validate(model, validation_dataloader):
    model.eval()
    criterion = nn.BCELoss().to(DEVICE)

    validation_loss, validation_accuracy = 0.0, 0.0

    with torch.set_grad_enabled(False):
        for inputs, lengths, labels in validation_dataloader:
            outputs = model(inputs.to(DEVICE), lengths).squeeze()

            accuracy_this_batch = accuracy_score(torch.round(outputs).detach().cpu().numpy(), labels)
            validation_accuracy += accuracy_this_batch

            loss = criterion(outputs.squeeze(), labels.to(DEVICE))
            validation_loss += loss.item()

    validation_loss /= len(validation_dataloader)
    validation_accuracy /= len(validation_dataloader)
    print('Validation loss={}\tvalidation accuracy={}'.format(validation_loss, validation_accuracy))

    return validation_loss, validation_accuracy


def test(model, test_dataloader):
    model.eval()

    accuracy = 0.0
    with torch.set_grad_enabled(False):
        for inputs, lengths, labels in tqdm(test_dataloader):
            outputs = model(inputs.to(DEVICE), lengths).squeeze()

            accuracy_this_batch = accuracy_score(torch.round(outputs).detach().cpu().numpy(), labels)
            accuracy += accuracy_this_batch

    print('Model accuracy = {}'.format(accuracy / len(test_dataloader)))


def train_and_validate(model, train_dl, valid_dl, epochs=30) -> pd.DataFrame:
    results = []
    best_loss = float('inf')
    for epoch in range(0, epochs, 1):
        train_loss, train_accuracy = train_one_epoch(model, train_dl, disable_pbar=True)
        validation_loss, validation_accuracy = validate(model, valid_dl)

        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'validation_loss': validation_loss,
            'validation_accuracy': validation_accuracy
        })

        # save the best model
        if validation_loss < best_loss:
            best_loss = validation_loss

            torch.save(model.state_dict(), os.path.join('output', 'tmp'))

    # revert model to best parameters
    model.load_state_dict(torch.load(os.path.join('output', 'tmp')))
    os.remove(os.path.join('output', 'tmp'))

    return pd.DataFrame(results)


def predict_sentence(sentence: str, model, encoder):
    df = pd.DataFrame([{
        'tweet': sentence,
        'label': 0,  # should be ignored, only for consistency
        'id': 1  # should be ignored, only for consistency
    }])

    preprocessor = DataframePreprocessor()
    df = preprocessor(df)

    inputs = torch.tensor([[encoder.encode(tk) for tk in df['tweet'].values[0]]]).long().to(DEVICE)
    lengths = torch.tensor([len(df['tweet'].values[0])]).long()

    outputs = model(inputs, lengths)

    return torch.round(outputs).detach().cpu().numpy()[0]
