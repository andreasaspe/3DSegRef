
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from model import UnetWithUncertainty, get_model_from_base_kwargs
import os
from torch.amp import autocast, GradScaler
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nnunet'))
from TotalSegmentator.utils.get_data_loader import get_data_loader
from uncertainty_eval import NDUncertaintyCalibration
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
            None, None, 100, 100
        )

        self.output_dir = training_kwargs['output_dir']
        print(self.output_dir)
        self.prepare_output_dir()

        self.model, self.new_modules = self.get_model(**model_kwargs)
        self.optimizer = self.setup_optimizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.train_loader, self.eval_loader = self.setup_data()
        self.num_iterations_per_epoch = training_kwargs.get('num_iterations_per_epoch', 250)
        self.num_val_iterations = training_kwargs.get('num_val_iterations', 50)
    
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

        return self.train_loader, self.eval_loader

    def prepare_output_dir(self, ):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self, **model_kwargs):

        model = get_model_from_base_kwargs(**model_kwargs)
        checkpoint_path = model_kwargs.pop('checkpoint_path', "")
        unmatched_modules = None
        
        if not checkpoint_path:
            print("Starting from untrained model")
        else:
            print("Starting from checkpoint {}".format(checkpoint_path))
            own_state_dict = model.state_dict()
            pretrained_state_dict = torch.load(checkpoint_path, weights_only=False)['network_weights']


            if any('decoder.encoder' in key for key in own_state_dict.keys()):
                new_state_dict = {}
                for key in pretrained_state_dict.keys():
                    new_key = key
                    if 'encoder' in key:
                        new_key = key.replace('encoder', 'decoder.encoder')
                    new_state_dict[new_key] = pretrained_state_dict[key]
                
            else:
                new_state_dict = pretrained_state_dict
            
            unmatched_modules = model.load_state_dict(pretrained_state_dict, strict=False)
        
        print(unmatched_modules)
        return model, unmatched_modules

    def train_iter(self, input_volume, target):

        input_volume = input_volume.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True).long()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            output = self.model(input_volume, targets=target)

        self.scaler.scale(output.loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return output

    def handle_various_input(self,elem):
        
        input_volume, target = elem['data'], elem['target']
        
        target = target[0].squeeze(1)
        if isinstance(input_volume, list):
            if not all(vol.shape[0] == 1 for vol in input_volume):
                input_volume = [vol.unsqueeze(0) for vol in input_volume]
            input_volume = torch.cat(input_volume, dim = 0)
        
        if isinstance(target, list):
            if not all(tar.shape[0] == 1 for tar in target):
                target = [tar.unsqueeze(0) for tar in target]
            target = torch.cat(target, dim = 0)
        
        return input_volume, target


    def train_one_epoch(self, epoch, last_performance = None):

        self.model.train()
        self.scaler = GradScaler(device='cuda')

        running_metrics = {}
    
        train_loader_bar = tqdm(range(self.num_iterations_per_epoch), desc = f'Training epoch {epoch}, performance: {last_performance}')
        for step in train_loader_bar:
            elem = next(self.train_loader)
            input_volume, target = self.handle_various_input(elem)
            output = self.train_iter(input_volume, target)

            for key, val in output.loss_decomp.items():
                if key not in running_metrics:
                    running_metrics[key] = []
                running_metrics[key].append(val)

            train_loader_bar.set_postfix({key: sum(val)/len(val) for key, val in running_metrics.items()})

    @torch.no_grad()
    def run_evaluation(self, eval_loader = None, basis_model_only = False):

        if eval_loader is None:
            eval_loader = self.eval_loader

        self.model.eval()
        num_eval_steps = len(eval_loader.generator._data.identifiers)
        eval_loader_bar = tqdm(range(num_eval_steps), desc = 'Running evaluation')

        results = {'dice_score': []}
        
        if basis_model_only:
            self.model.basis_model_only()
        
        for step in eval_loader_bar:
            elem = next(eval_loader)
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')
            
            if step == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
            
            dice = dice_score(output.mu, target)
            results['dice_score'].append(dice)

        metrics = {key: sum(val) / len(val) for key, val in results.items()}

        self.model.basis_model_only(False)
        return metrics


    def save_model(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch)))
        print("Saved model to", os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch)))

    def train(self, ):

        performance = {'dice_score': -1}
        best_performance = 0
        self.model.to(self.device)
        basis_metrics = self.run_evaluation(basis_model_only=True)
        print(basis_metrics)

        for epoch in range(self.training_kwargs['num_epochs']):
            self.train_one_epoch(epoch, last_performance = performance)
            performance = self.run_evaluation()
            if performance['dice_score'].item() > best_performance:
                self.save_model(epoch)
                best_performance = performance['dice_score'].item()


if __name__ == '__main__':

    model_kwargs = {
        'checkpoint_path': '/scratch/pjtka/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': dict(),
        'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl'
    }

    training_kwargs = {
        'num_epochs': 10,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': '/scratch/pjtka/ndseg_output',
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 130
    }


    trainer = Trainer(model_kwargs, training_kwargs)
    trainer.train()

    breakpoint()














