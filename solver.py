import torch

from torch import nn
from torch.utils import data


class Solver():

    def __init__(self, optim=torch.optim.Adam, optim_args={'lr': 1e-3}, loss_func=torch.nn.CrossEntropyLoss):
        self.optim = optim
        self.optim_args = optim_args
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []

    def train(self, device, model: nn.Module, train_loader: data.DataLoader, val_loader: data.DataLoader, num_epochs=10,
              log_nth=0):
        self._reset_histories()

        model.to(device)
        optimizer = self.optim(model.parameters(), **self.optim_args)
        loss_func = self.loss_func()

        for epoch in range(num_epochs):
            model.train()

            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (input, target) in enumerate(train_loader):
                # import pdb; pdb.set_trace()
                input = input.to(device).float()
                target = target.to(device)

                score = model(input)
                loss = loss_func(score, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if log_nth != 0 and (i + 1) % log_nth == 0:
                    print("Iteration: {}/{}, Loss: {}".format(i + 1, num_epochs * len(train_loader),
                                                              loss.item()))

                prediction = torch.argmax(score, dim=1)
                acc = float(torch.sum(prediction == target)) / target.shape[0]

                epoch_loss += loss.item()
                epoch_acc += acc

            epoch_loss /= len(train_loader)
            epoch_acc /= len(train_loader)
            self.train_loss_history.append(epoch_loss)
            self.train_acc_history.append(epoch_acc)
            print("Epoch: {}/{}, [Train] Loss: {}, Accuracy: {}".format(epoch + 1, num_epochs, epoch_loss, epoch_acc))

            model.eval()

            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (input, target) in enumerate(val_loader):
                input = input.to(device).float()
                target = target.to(device)

                score = model(input)
                loss = loss_func(score, target)

                prediction = torch.argmax(score, dim=1)
                acc = float(torch.sum(prediction == target)) / target.shape[0]

                # print(target)
                # print(prediction)
                # print('')

                epoch_loss += loss.item()
                epoch_acc += acc

            epoch_loss /= len(val_loader)
            epoch_acc /= len(val_loader)
            self.val_loss_history.append(epoch_loss)
            self.val_acc_history.append(epoch_acc)
            print("Epoch: {}/{}, [Valid] Loss: {}, Accuracy: {}".format(epoch + 1, num_epochs, epoch_loss, epoch_acc))
