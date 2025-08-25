from http.cookiejar import unmatched

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from model import UnetWithUncertainty, get_model_from_base_kwargs
import os
from torch.cuda.amp import autocast, GradScaler
import sys
sys.path.append(r'C:\Users\pjtka\PycharmProjects\ConceptBottleneckP2\3DSegRef\nnunet')
#from TotalSegmentator.utils.get_data_loader import get_data_loader
from uncertainty_eval import NDUncertaintyCalibration


def dice_score(prediction, target, smooth = 1e-6, reduction = 'None'):

    prediction = prediction.argmax(1)
    intersection = torch.sum(prediction * target)
    out = (2. * intersection + smooth) / (torch.sum(target) + torch.sum(prediction) + smooth)
    return out.detach().cpu()

class Trainer(object):
    def __init__(self, model_kwargs, training_kwargs):

        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

        self.train_loader, self.eval_loader, self.num_batches_train, self.num_batches_eval = (
            None, None, None, None
        )

        self.output_dir = training_kwargs['output_dir']
        self.prepare_output_dir()

        self.model, self.new_modules = self.get_model(**model_kwargs)
        self.optimizer = self.setup_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.train_loader, self.eval_loader = self.setup_data()

    def setup_optimizer(self,):
        """
        Can add potential functionality for different learning rates here
        :return:
        """
        lr = self.training_kwargs['lr']
        weight_decay = self.training_kwargs['weight_decay']
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def setup_data(self, ):
        self.train_loader, self.eval_loader = get_data_loader()

        self.num_batches_train = len(self.train_loader)
        self.num_batches_eval = len(self.eval_loader)

        return self.train_loader, self.eval_loader

    def prepare_output_dir(self, ):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self, **model_kwargs):

        model = get_model_from_base_kwargs(**model_kwargs)
        checkpoint_path = model_kwargs.pop('checkpoint_path', "")
        unmatched_modules = None
        if checkpoint_path:
            print("Starting from untrained model")
        else:
            print("Starting from checkpoint {}".format(checkpoint_path))
            unmatched_modules = model.load_state_dict(torch.load(checkpoint_path), strict=False)

        return model, unmatched_modules

    def train_iter(self, input_volume, target):

        input_volume = input_volume.to(self.device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp):
            output = self.model(input_volume, target=target)

        self.scaler.scale(output.loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return output

    def train_one_epoch(self, epoch, last_performance = None):

        self.model.train()
        self.scaler = GradScaler(enabled=amp)

        running_metrics = {}
        train_loader_bar = tqdm(self.train_loader, desc = f'Training epoch {epoch}, performance: {last_performance}')
        for input_volume, target in train_loader_bar:
            output = self.train_iter(input_volume, target)

            for key, val in output.loss_attributes.items():
                if key not in running_metrics:
                    running_metrics[key] = []
                running_metrics[key].append(val)

            train_loader_bar.set_postfix(running_metrics)

    @torch.no_grad()
    def run_evaluation(self, eval_loader = None):

        if eval_loader is None:
            eval_loader = self.eval_loader

        self.model.eval()
        eval_loader_bar = tqdm(eval_loader, desc = 'Running evaluation')

        results = []

        for input_volume, target in eval_loader_bar:
            output = self.model.inference(input_volume)
            dice = dice_score(output, target)
            results.append(dice)

        metrics = {'dice': sum(results) / len(results)}

        return metrics


    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch)))
        print("Saved model to", os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch)))

    def train(self, ):

        performance = {'dice': -1}
        best_performance = 0
        for epoch in range(self.training_kwargs['epochs']):
            self.train_one_epoch(epoch, last_performance = performance)
            performance = self.run_evaluation()
            if performance['dice'] > best_performance:
                self.save_model(epoch)


if __name__ == '__main__':

    model_kwargs = {
        'checkpoint_path': r"C:\Users\pjtka\Documents\3d_uncertainty\test_loading.pth",
        'loss_kwargs': dict(),
    }

    training_kwargs = {
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': r'C:\Users\pjtka\Documents\3d_uncertainty'
    }


    trainer = Trainer(model_kwargs, training_kwargs)
    breakpoint()














