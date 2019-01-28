import torch
import torch.nn as nn
import torch.optim as onn


class FFN(nn.Module):

    def __init__(self, n_features, hidden, n_classes):
        nn.Module.__init__(self)
        self.classifier = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
            nn.Softmax(dim=0)
        )
        self.error_func = nn.BCELoss()
        self.optimizer = onn.RMSprop(params=self.parameters(), lr=0.0001)

    def forward(self, input_vectors):
        return self.classifier(input_vectors)

    def fit(self, input_vectors, targets, batch_size=50, epochs=10, verbose=False):
        self.train()
        self.zero_grad()
        input_vectors = torch.Tensor(input_vectors.astype(float))
        targets = torch.Tensor(targets.astype(float))
        for epoch in range(epochs):
            for batch_ix in range(0, input_vectors.size(0), batch_size):
                batch_in = input_vectors[batch_ix: batch_ix + batch_size]
                batch_targets = targets[batch_ix: batch_ix + batch_size]
                output = self.forward(batch_in)
                error = self.error_func(output, batch_targets)
                error.backward()
                self.optimizer.step()
                if verbose:
                    print('Loss: {}'.format(error.item()))

    def predict_proba(self, input_vectors):
        self.eval()
        input_vectors = torch.Tensor(input_vectors.astype(float))
        probas = self.classifier(input_vectors)
        return probas.detach().numpy().squeeze()

    def predict(self, input_vectors):
        self.eval()
        input_vectors = torch.Tensor(input_vectors.astype(float))
        probas = self.classifier(input_vectors)
        predictions = (probas <= 0.5)
        return predictions.detach().numpy().squeeze()
