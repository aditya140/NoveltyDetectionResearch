# %%
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# %%

train_set = torchvision.datasets.FashionMNIST(
    "./data", download=True, transform=transforms.Compose([transforms.ToTensor()])
)
test_set = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
)

# %%
import torch.nn.functional as F
from torch.optim import Adam


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2
        )
        self.conv1_s = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=5, stride=2, padding=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )
        self.conv2_s = nn.Conv2d(
            in_channels=6, out_channels=6, kernel_size=3, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1
        )
        self.conv3_s = nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(160, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1_s(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2_s(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3_s(x))

        x = self.flatten(x)
        x = self.fc1(x)
        x = F.softmax(x)

        return x


# %%
batch_size = 64
learning_rate = 1e-3
epochs = 20


from torch.utils.data import TensorDataset, DataLoader

X = train_set.data.numpy().reshape(train_set.data.shape[0], -1)
y = train_set.targets.numpy()

X_test = test_set.data.numpy().reshape(test_set.data.shape[0], -1)
y_test = test_set.targets.numpy()


x_train_tensor = torch.tensor(X.reshape(-1, 1, 28, 28).astype(np.float32))
x_test_tensor = torch.tensor(X_test.reshape(-1, 1, 28, 28).astype(np.float32))

y_train_tensor = torch.tensor(y.astype(np.long))
y_test_tensor = torch.tensor(y_test.astype(np.long))


train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size)

test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
simple_cnn = SimpleCNN().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(simple_cnn.parameters(), lr=learning_rate)

# %%
for epoch in range(epochs):
    simple_cnn.train()
    train_samples_count = 0
    true_train_samples_count = 0
    running_loss = 0

    for batch in train_loader:
        x_data = batch[0].cuda()
        y_data = batch[1].cuda()

        y_pred = simple_cnn(x_data)
        loss = criterion(y_pred, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        y_pred = y_pred.argmax(dim=1, keepdim=False)
        true_classified = (y_pred == y_data).sum().item()
        true_train_samples_count += true_classified
        train_samples_count += len(x_data)

    train_accuracy = true_train_samples_count / train_samples_count
    print(f"[{epoch}] train loss: {running_loss}, accuracy: {round(train_accuracy, 4)}")

    simple_cnn.eval()
    test_samples_count = 0
    true_test_samples_count = 0
    running_loss = 0

    for batch in test_loader:
        x_data = batch[0].cuda()
        y_data = batch[1].cuda()

        y_pred = simple_cnn(x_data)
        loss = criterion(y_pred, y_data)

        loss.backward()

        running_loss += loss.item()

        y_pred = y_pred.argmax(dim=1, keepdim=False)
        true_classified = (y_pred == y_data).sum().item()
        true_test_samples_count += true_classified
        test_samples_count += len(x_data)

    test_accuracy = true_test_samples_count / test_samples_count
    print(f"[{epoch}] test loss: {running_loss}, accuracy: {round(test_accuracy, 4)}")
# %%
"""
Ensemble
"""
import sklearn
from sklearn.ensemble import BaggingClassifier, VotingClassifier

# %%
class TorchEnsembleModel(sklearn.base.BaseEstimator):
    def __init__(
        self,
        net_type,
        net_params,
        optim_type,
        optim_params,
        loss_fn,
        input_shape,
        accuracy_tol=0.02,
        batch_size=32,
        tol_epochs=10,
        cuda=True,
    ):
        self.net_type = net_type
        self.net_params = net_params
        self.optim_type = optim_type
        self.optim_params = optim_params
        self.loss_fn = loss_fn

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.accuracy_tol = accuracy_tol
        self.tol_epochs = tol_epochs
        self.cuda = cuda

    def fit(self, X, y):
        self.net = self.net_type(**self.net_params)
        if self.cuda:
            self.net = self.net.cuda()
        self.optim = self.optim_type(self.net.parameters(), **self.optim_params)

        uniq_classes = np.sort(np.unique(y))
        self.classes_ = uniq_classes

        X = X.reshape(-1, *self.input_shape)
        x_tensor = torch.tensor(X.astype(np.float32))
        y_tensor = torch.tensor(y.astype(np.long))
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        last_accuracies = []
        epoch = 0
        keep_training = True
        while keep_training:
            self.net.train()
            train_samples_count = 0
            true_train_samples_count = 0
            epoch += 1
            print(f"Epoch {epoch}")
            for batch in train_loader:
                x_data, y_data = batch[0], batch[1]
                if self.cuda:
                    x_data = x_data.cuda()
                    y_data = y_data.cuda()

                y_pred = self.net(x_data)
                loss = self.loss_fn(y_pred, y_data)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                y_pred = y_pred.argmax(dim=1, keepdim=False)
                true_classified = (y_pred == y_data).sum().item()
                true_train_samples_count += true_classified
                train_samples_count += len(x_data)

            train_accuracy = true_train_samples_count / train_samples_count
            last_accuracies.append(train_accuracy)

            if len(last_accuracies) > self.tol_epochs:
                last_accuracies.pop(0)

            if epoch >= self.tol_epochs:
                keep_training = False

            if len(last_accuracies) == self.tol_epochs:
                accuracy_difference = max(last_accuracies) - min(last_accuracies)
                if accuracy_difference <= self.accuracy_tol:
                    keep_training = False

    def predict_proba(self, X, y=None):
        X = X.reshape(-1, *self.input_shape)
        x_tensor = torch.tensor(X.astype(np.float32))
        if y:
            y_tensor = torch.tensor(y.astype(np.long))
        else:
            y_tensor = torch.zeros(len(X), dtype=torch.long)

        test_dataset = TensorDataset(x_tensor, y_tensor)
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        self.net.eval()
        predictions = []
        for batch in test_loader:
            x_data, y_data = batch[0], batch[1]
            if self.cuda:
                x_data = x_data.cuda()
                y_data = y_data.cuda()

            y_pred = self.net(x_data)

            predictions.append(y_pred.detach().cpu().numpy())

        predictions = np.concatenate(predictions)
        return predictions

    def predict(self, X, y=None):
        predictions = self.predict_proba(X, y)
        predictions = predictions.argmax(axis=1)
        return predictions


# %%
base_model = PytorchModel(
    net_type=SimpleCNN,
    net_params=dict(),
    optim_type=Adam,
    optim_params={"lr": 1e-3},
    loss_fn=nn.CrossEntropyLoss(),
    input_shape=(1, 28, 28),
    accuracy_tol=0.02,
    tol_epochs=10,
    cuda=True,
)
# %%
base_model.fit(X, y)

# %%
preds = base_model.predict(X_test)

# %%
true_classified = (preds == y_test).sum()
test_accuracy = true_classified / len(y_test)
print(f"Test accuracy: {test_accuracy}")
# %%
meta_classifier = BaggingClassifier(base_estimator=base_model, n_estimators=10)

# %%
meta_classifier.fit(X, y)

# %%
meta_classifier.score(X_test, y_test)

# %%
