from ignite.engines import Events
from tensorboardX import SummaryWriter
import torch

from train import create_supervised_evaluator, create_supervised_trainer


class Runner(object):

    def __init__(self, model, loss_fn, metrics, optimizer, y_to_score, pred_to_score, device, log_dir):
        if device != -1:
            with torch.cuda.device(device):
                self.model = model.cuda()
        else:
            self.model = model

        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.y_to_score = y_to_score
        self.pred_to_score = pred_to_score
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self, epochs, train_loader, val_loader, test_loader, log_interval):
        cuda = self.device != -1
        with torch.cuda.device(self.device):
            trainer = create_supervised_trainer(self.model, self.optimizer, self.loss_fn, cuda=cuda)
            evaluator = create_supervised_evaluator(self.model, metrics=self.metrics, y_to_score=self.y_to_score, pred_to_score=self.pred_to_score, cuda=cuda)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(train_loader) + 1
            if iter % log_interval == 0:
                print("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                      "".format(engine.state.epoch, iter, len(train_loader), engine.state.output))
                self.writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            state_metrics = evaluator.state.metrics

            state_metric_keys = list(self.metrics.keys())
            state_metric_vals = [state_metrics[k] for k in state_metric_keys]
            format_str = 'Validation Results - Epoch: {} ' + ' '.join([k + ': {:.4f}' for k in state_metric_keys])
            print(format_str.format(*([engine.state.epoch] + state_metric_vals)))
            for i, k in enumerate(state_metric_keys):
                self.writer.add_scalar(f'dev/{k}', state_metric_vals[i], engine.state.epoch)

        @trainer.on(Events.COMPLETED)
        def log_test_results(engine):
            evaluator.run(test_loader)
            state_metrics = evaluator.state.metrics

            state_metric_keys = list(self.metrics.keys())
            state_metric_vals = [state_metrics[k] for k in state_metric_keys]
            format_str = 'Test Results - Epoch: {} ' + ' '.join([k + ': {:.4f}' for k in state_metric_keys])
            print(format_str.format(*([engine.state.epoch] + state_metric_vals)))
            for i, k in enumerate(state_metric_keys):
                self.writer.add_scalar(f'test/{k}', state_metric_vals[i], engine.state.epoch)

        trainer.run(train_loader, max_epochs=epochs)

        self.writer.close()
