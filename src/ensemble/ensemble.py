import torch
import torch.nn as nn
import torch.nn.functional as F

from torchensemble._base import BaseModule, torchensemble_model_doc
from torchensemble.utils import io
from torchensemble.utils import set_module
from torchensemble.utils import operator as op


import warnings
from joblib import Parallel, delayed


from torchensemble import FusionClassifier, VotingClassifier


@torchensemble_model_doc("""Implementation on the FusionClassifier.""", "model")
class FusionClassifier_novelty(FusionClassifier):
    def _forward(self, x, y):
        """
        Implementation on the internal data forwarding in FusionClassifier.
        """
        # Average
        outputs = [estimator(x, y) for estimator in self.estimators_]
        output = op.average(outputs)

        return output

    @torchensemble_model_doc(
        """Implementation on the data forwarding in FusionClassifier.""",
        "classifier_forward",
    )
    def forward(self, x, y):
        output = self._forward(x, y)
        proba = F.softmax(output, dim=1)

        return proba

    @torchensemble_model_doc(
        """Implementation on the training stage of FusionClassifier.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        # Instantiate base estimators and set attributes
        for _ in range(self.n_estimators):
            self.estimators_.append(self._make_estimator())
        self._validate_parameters(epochs, log_interval)
        self.n_outputs = 2
        optimizer = set_module.set_optimizer(
            self, self.optimizer_name, **self.optimizer_args
        )

        # Set the scheduler if `set_scheduler` was called before
        if self.use_scheduler_:
            self.scheduler_ = set_module.set_scheduler(
                optimizer, self.scheduler_name, **self.scheduler_args
            )

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        # Training loop
        for epoch in range(epochs):
            self.train()
            for batch_idx, (batch) in enumerate(train_loader):

                source, target, label = (
                    batch.source.to(self.device),
                    batch.target.to(self.device),
                    batch.label.to(self.device),
                )

                # data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self._forward(source, target)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                # Print training status
                if batch_idx % log_interval == 0:
                    with torch.no_grad():
                        _, predicted = torch.max(output.data, 1)
                        correct = (predicted == label).sum().item()

                        msg = (
                            "Epoch: {:03d} | Batch: {:03d} | Loss:"
                            " {:.5f} | Correct: {:d}/{:d}"
                        )
                        self.logger.info(
                            msg.format(epoch, batch_idx, loss, correct, label.size(0))
                        )

            # Validation
            if test_loader:
                self.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for _, (batch) in enumerate(test_loader):

                        source, target, label = (
                            batch.source.to(self.device),
                            batch.target.to(self.device),
                            batch.label.to(self.device),
                        )

                        output = self.forward(source, target)
                        _, predicted = torch.max(output.data, 1)
                        correct += (predicted == label).sum().item()
                        total += label.size(0)
                    acc = 100 * correct / total

                    if acc > best_acc:
                        best_acc = acc
                        if save_model:
                            io.save(self, save_dir, self.logger)

                    msg = (
                        "Epoch: {:03d} | Validation Acc: {:.3f}"
                        " % | Historical Best: {:.3f} %"
                    )
                    self.logger.info(msg.format(epoch, acc, best_acc))

            # Update the scheduler
            if hasattr(self, "scheduler_"):
                self.scheduler_.step()

        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of FusionClassifier.""",
        "classifier_predict",
    )
    def predict(self, test_loader):
        self.eval()
        correct = 0
        total = 0

        for _, (batch) in enumerate(test_loader):
            source, target, label = (
                batch.source.to(self.device),
                batch.target.to(self.device),
                batch.label.to(self.device),
            )
            output = self.forward(source, target)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        acc = 100 * correct / total

        return acc


def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
):
    """
    Private function used to fit base estimators in parallel.
    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """

    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, (batch) in enumerate(train_loader):

        source, target, label = (
            batch.source.to(device),
            batch.target.to(device),
            batch.label.to(device),
        )

        batch_size = label.size(0)

        optimizer.zero_grad()
        output = estimator(source, target)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == label).sum().item()

                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f} | Correct: {:d}/{:d}"
                )
                print(msg.format(idx, epoch, batch_idx, loss, correct, batch_size))
            # Regression
            else:
                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f}"
                )
                print(msg.format(idx, epoch, batch_idx, loss))

    return estimator, optimizer


@torchensemble_model_doc("""Implementation on the VotingClassifier.""", "model")
class VotingClassifier_novelty(VotingClassifier):
    @torchensemble_model_doc(
        """Implementation on the data forwarding in VotingClassifier.""",
        "classifier_forward",
    )
    def forward(self, x, y):
        # Take the average over class distributions from all base estimators.
        outputs = [F.softmax(estimator(x, y), dim=1) for estimator in self.estimators_]
        proba = op.average(outputs)

        return proba

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingClassifier.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = 2

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = []
        for _ in range(self.n_estimators):
            estimators.append(self._make_estimator())

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Utils
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0

        # Internal helper function on pesudo forward
        def _forward(estimators, x, y):
            outputs = [F.softmax(estimator(x, y), dim=1) for estimator in estimators]
            proba = op.average(outputs)

            return proba

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        True,
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers = [], []
                for estimator, optimizer in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for _, batch in enumerate(test_loader):
                            source, target, label = (
                                batch.source.to(self.device),
                                batch.target.to(self.device),
                                batch.label.to(self.device),
                            )

                            output = _forward(estimators, source, target)
                            _, predicted = torch.max(output.data, 1)
                            correct += (predicted == label).sum().item()
                            total += label.size(0)
                        acc = 100 * correct / total

                        if acc > best_acc:
                            best_acc = acc
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = (
                            "Epoch: {:03d} | Validation Acc: {:.3f}"
                            " % | Historical Best: {:.3f} %"
                        )
                        self.logger.info(msg.format(epoch, acc, best_acc))

                # Update the scheduler
                with warnings.catch_warnings():

                    # UserWarning raised by PyTorch is ignored because
                    # scheduler does not have a real effect on the optimizer.
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        scheduler_.step()

        self.estimators_ = nn.ModuleList()
        self.estimators_.extend(estimators)
        if save_model and not test_loader:
            io.save(self, save_dir, self.logger)

    @torchensemble_model_doc(
        """Implementation on the evaluating stage of VotingClassifier.""",
        "classifier_predict",
    )
    def predict(self, test_loader):
        self.eval()
        estimators = [i for i in self.estimators_]

        def _forward(estimators, x, y):
            outputs = [F.softmax(estimator(x, y), dim=1) for estimator in estimators]
            proba = op.average(outputs)
            return proba

        with torch.no_grad():
            correct = 0
            total = 0
            for _, batch in enumerate(test_loader):
                source, target, label = (
                    batch.source.to(self.device),
                    batch.target.to(self.device),
                    batch.label.to(self.device),
                )

                output = _forward(estimators, source, target)
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            acc = 100 * correct / total
        return acc
