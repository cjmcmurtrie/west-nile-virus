import torch
import numpy as np
import torch.nn as nn
import torch.optim as onn


torch.manual_seed(1)


class LSTMClassifier(nn.Module):

    def __init__(self, n_features, hidden, n_classes):
        nn.Module.__init__(self)
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)   # , bidirectional=True
        self.classifier = nn.Sequential(
            nn.Linear(hidden, n_classes),
            nn.Sigmoid()
        )
        self.error_func = nn.BCELoss()
        self.optimizer = onn.RMSprop(params=self.parameters(), lr=0.0001)
        self.cuda()

    def forward(self, input_vectors):
        out, (h, c) = self.rnn(input_vectors)
        return self.classifier(out).squeeze(2)

    def fit(self, input_vectors, targets, epochs=20, verbose=False):
        self.train()
        self.zero_grad()
        input_vectors = [torch.Tensor(npa).unsqueeze(0).cuda() for npa in input_vectors]
        targets = [torch.Tensor(npa).cuda() for npa in targets]
        for epoch in range(epochs):
            ix = torch.randperm(len(input_vectors))
            input_vectors = [input_vectors[i] for i in ix]
            targets = [targets[i] for i in ix]
            for i, (seq, tar) in enumerate(zip(input_vectors, targets)):
                output = self.forward(seq)
                error = self.error_func(output.squeeze(), tar)
                error.backward()
                self.optimizer.step()
                if verbose:
                    print('Epoch: {}. Batch: {}. Error: {}'.format(epoch, i, error.item()))

    def predict_proba(self, input_sequences):
        self.eval()
        probas = []
        for sequence in input_sequences:
            seq = torch.Tensor(sequence).cuda()
            proba = self.forward(seq.unsqueeze(0))
            probas.append(proba.squeeze(0))
        return torch.cat(probas, 0).detach().cpu().numpy()

    def predict(self, input_vectors):
        self.eval()
        input_vectors = torch.Tensor(input_vectors).cuda()
        probas = self.classifier(input_vectors)
        predictions = (probas <= 0.5)
        return predictions.detach().cpu().numpy().squeeze()


if __name__ == '__main__':
    input_data = np.random.rand(1, 30, 50)
    model = LSTMClassifier(50, 20, 1)
    print(input_data)
    print(model.forward(torch.Tensor(input_data).cuda()))
