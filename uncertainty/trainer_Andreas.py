
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from model_Andreas import UnetWithUncertainty, get_model_from_base_kwargs
import os
from torch.amp import autocast, GradScaler
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nnunet'))
from TotalSegmentator.utils.get_data_loader import get_data_loader
from uncertainty_eval import NDUncertaintyCalibration
import warnings
import pickle
import numpy as np
from argparse import ArgumentParser, Namespace
import ast
from dataloaders import HackyEvalLoader
import SimpleITK as sitk
warnings.filterwarnings('ignore', category=UserWarning)

torch.set_num_interop_threads(1)
def dice_score(prediction, target, smooth = 1e-6, reduction = 'None'):

    prediction = prediction.argmax(1)
    intersection = torch.sum(prediction * target)
    out = (2. * intersection + smooth) / (torch.sum(target) + torch.sum(prediction) + smooth)
    return out.detach().cpu()

class Logger:
    def __init__(self, write_path, experiment_name = ""):

        self.write_path = write_path
        os.makedirs(write_path, exist_ok=True)

        self.save_path = os.path.join(self.write_path, f"{experiment_name}_results.pkl")
        self.latest_key = None

    @staticmethod
    def handle_results_dtype(results):
        new_results = {}
        for key, val in results.items():
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            
            if isinstance(val, np.ndarray):
                if np.prod(val.shape) == 1:
                    val = val.item()

            new_results[key] = val 

        return new_results    

    def write(self, results: dict, key: str):
        
        
        if os.path.isfile(self.save_path):
            results_pcl = pickle.load(open(self.save_path, 'rb'))
        else:
            results_pcl = {}
        
        results = self.handle_results_dtype(results)
        if key not in results_pcl:
            results_pcl[key] = []
        results_pcl[key].append(results)

        with open(self.save_path, 'wb') as handle:
            pickle.dump(results_pcl, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        print("Wrote results to", self.save_path)

class PerformanceKeeper:
    def __init__(self, performances):
        self.performances = performances
    
    def __add__(self, other):
        new_performances = {key: val + other.performances[key] for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)
    
    def __sub__(self, other):
        new_performances = {key: val - other.performances[key] for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)
    
    def __mul__(self, scalar):
        new_performances = {key: val * scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __div__(self, scalar):
        new_performances = {key: val / scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __truediv__(self, scalar):
        new_performances = {key: val /scalar for key, val in self.performances.items()}
        return PerformanceKeeper(new_performances)

    def __repr__(self):
        return repr(self.performances)
    
    def __str__(self):
        return str(self.performances)
    

class PerformanceHolder:
    def __init__(self,):

        self.performances = {}

    def update(self, key, performance):
        self.performances[key] = PerformanceKeeper(performance)
    
    def to_list(self, keys = None):
        keys = self.performances.keys() if keys is None else keys
        values = [self.performances[key] for key in keys]
        return values
        
    def mean(self, keys = None):
        values = self.to_list(keys = keys)
        mean = values[0]
        for elem in values[1:]:
            mean = mean + elem
        return mean * 1/len(values)
    
    def mean_with_other(self, other):
        combined_keys = set(self.performances.keys()).intersection(set(other.performances.keys()))
        return self.mean(keys = combined_keys)
    

        
class Trainer(object):
    def __init__(self, model_kwargs, training_kwargs):

        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs
        self.model_kwargs['loss_kwargs'] = training_kwargs['loss_kwargs']
        self.get_eval_loader_from_saved = training_kwargs.get('eval_loader_data_path', "")

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
        self.last_saved_model_path = ""
        
        self.basis_model_results_keeper = None
        self.trained_models_results_keeper = None
        self.last_performance_diff = None
        self.gradient_accumulation = self.training_kwargs.get('gradient_accumulation', False)


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

        if self.get_eval_loader_from_saved:
            self.eval_loader = HackyEvalLoader(self.get_eval_loader_from_saved)

        return self.train_loader, self.eval_loader

    def prepare_output_dir(self, ):
        os.makedirs(self.output_dir, exist_ok=True)

    def get_model(self, **model_kwargs):
        checkpoint_path = model_kwargs.pop('checkpoint_path', "")
        
        model = get_model_from_base_kwargs(**model_kwargs)
        unmatched_modules = None
        
        if not checkpoint_path:
            print("Starting from untrained model")
        else:
            print("Starting from checkpoint {}".format(checkpoint_path))
            own_state_dict = model.state_dict()
            if self.model_kwargs['basic']: # Raw nn-unet output. Has more keys than just network_weights and thus we need to specify
                pretrained_state_dict = torch.load(checkpoint_path, weights_only=False)['network_weights']
            else: # After stocastic training
                pretrained_state_dict = torch.load(checkpoint_path, weights_only=False)


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

        #with autocast(device_type='cuda'):
        scaler = None
        if self.gradient_accumulation:
            scaler = self.scaler
        
        output = self.model(input_volume, targets=target, scaler = scaler)
        
        if scaler is None:
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
    
        train_loader_bar = tqdm(range(self.num_iterations_per_epoch), desc = f'Training epoch {epoch}, performance diff: {self.last_performance_diff}')
        for step in train_loader_bar:
            elem = next(self.train_loader)
            input_volume, target = self.handle_various_input(elem)
            output = self.train_iter(input_volume, target)

            for key, val in output.loss_decomp.items():
                if key not in running_metrics:
                    running_metrics[key] = []
                running_metrics[key].append(val)

            train_loader_bar.set_postfix({key: f"{sum(val)/len(val):0.3f}" for key, val in running_metrics.items()})
        
        print({key: f"{sum(val)/len(val):0.3f}" for key, val in running_metrics.items()})
    
    @torch.no_grad()
    def run_evaluation_new(self, eval_loader = None, basis_model_only = False):

        if eval_loader is None:
            eval_loader = self.eval_loader


        if not isinstance(eval_loader, HackyEvalLoader):
            return self.run_evaluation(eval_loader=eval_loader, basis_model_only=basis_model_only)
    
        self.model.eval()
        num_eval_steps = len(eval_loader)
        eval_loader_bar = tqdm(desc = 'Running evaluation', total = num_eval_steps)
        performance_holder = PerformanceHolder()

        if basis_model_only:
            self.model.basis_model_only()
        
        ce_forward = nn.CrossEntropyLoss()
        
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            self.model.decoder.cov_weighting_head.track_chosen_indices()

        results = {'dice_score': [], 'cross_entropy_mean': []}
        for i in range(num_eval_steps):
            elem = eval_loader[i]
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')

            if i == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
        
            cross_entropy = ce_forward(output.mu, target)
            
            dice = dice_score(output.mu, target)
            results['dice_score'].append(dice)
            results['cross_entropy_mean'].append(cross_entropy.item())
            performance = {key: val[-1] for key, val in results.items()}
            performance_holder.update(elem['keys'][0].item(), performance)
            eval_loader_bar.update(1)
        
        
        metrics = {key: sum(val) / len(val) for key, val in results.items()}
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            metrics['basis_indices'] = {
                val: self.model.decoder.cov_weighting_head.chosen_indices.count(val) for val in np.unique(
                    self.model.decoder.cov_weighting_head.chosen_indices
                )
            }
            self.model.decoder.cov_weighting_head.track_chosen_indices(False)
        
            print(metrics['basis_indices'])
        if basis_model_only:
            self.basis_model_results_keeper = performance_holder
        else:
            self.trained_models_results_keeper = performance_holder
        self.model.basis_model_only(False)
        
        return metrics, performance_holder.mean()
    
    
    
    @torch.no_grad()
    def run_evaluation_new_to_get_variance(self, eval_loader = None, basis_model_only = False):

        if eval_loader is None:
            eval_loader = self.eval_loader
        
        
        self.model.to(self.device)

        if not isinstance(eval_loader, HackyEvalLoader):
            return self.run_evaluation(eval_loader=eval_loader, basis_model_only=basis_model_only)
    
        self.model.eval()
        num_eval_steps = len(eval_loader)
        eval_loader_bar = tqdm(desc = 'Running evaluation', total = num_eval_steps)
        performance_holder = PerformanceHolder()

        if basis_model_only:
            self.model.basis_model_only()
        
        ce_forward = nn.CrossEntropyLoss()
        
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            self.model.decoder.cov_weighting_head.track_chosen_indices()

        results = {'dice_score': [], 'cross_entropy_mean': []}
        for i in range(num_eval_steps):
            elem = eval_loader[i]
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')

            savepath_root = "/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/pred"
            image_root_path = "/home/awias/data/nnUNet/nnUNet_raw/Dataset004_TotalSegmentatorPancreas/imagesTr"
            subject = elem['keys'].item()
            # Assuming output.variance is a tensor of shape [1, 2, 224, 160, 192]
            if output.variance is not None:
                imgpath = os.path.join(image_root_path, subject + "_0000.nii.gz")
                print("Yay, variance is not None")
                savepath = os.path.join(savepath_root, "variance")
                os.makedirs(savepath, exist_ok=True)
                variance = output.variance.detach().cpu().numpy()  # shape: (1, 2, 224, 160, 192)
                # for class_idx in range(variance.shape[1]):
                #     # Remove batch dimension
                #     class_variance = variance[0, class_idx]
                #     class_variance = (class_variance - class_variance.min()) / (class_variance.max() - class_variance.min() + 1e-8) * 100
                #     nifti_img = nib.Nifti1Image(class_variance, affine=np.eye(4))
                #     nib.save(nifti_img, os.path.join(savepath, f"variance_class_{class_idx}_sample_{i}.nii.gz"))

                # Remove batch dimension
                variance_background = variance[0, 0]
                variance_foreground = variance[0, 1]
                nifti_variance_foreground = nib.Nifti1Image(variance_foreground, affine=np.eye(4))
                nifti_variance_background = nib.Nifti1Image(variance_background, affine=np.eye(4))
                nib.save(nifti_variance_foreground, os.path.join(savepath, f"variance-foreground_{subject}_stochastic.nii.gz"))
                nib.save(nifti_variance_background, os.path.join(savepath, f"variance-background_{subject}_stochastic.nii.gz"))

                entropy = output.entropy.detach().cpu().numpy().squeeze(0).astype(np.float32)  # shape: (224, 160, 192)
                prediction_np = output.mu.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                target_np = target.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                input_np = input_volume.squeeze(0,1).detach().cpu().numpy().astype(np.float32)
                logits_np = output.mu.detach().cpu().numpy().squeeze(0).astype(np.float32)
                logits_background_np = logits_np[0]
                logits_foreground_np = logits_np[1]
                

                nifti_input = nib.Nifti1Image(input_np, affine=np.eye(4))
                nib.save(nifti_input, os.path.join(savepath, f"input_{subject}_stochastic.nii.gz"))
                
                nifti_pred = nib.Nifti1Image(prediction_np, affine=np.eye(4))
                nib.save(nifti_pred, os.path.join(savepath, f"prediction_{subject}_stochastic.nii.gz"))

                nifti_target = nib.Nifti1Image(target_np, affine=np.eye(4))
                nib.save(nifti_target, os.path.join(savepath, f"target_{subject}_stochastic.nii.gz"))

                nifti_entropy = nib.Nifti1Image(entropy, affine=np.eye(4))
                nib.save(nifti_entropy, os.path.join(savepath, f"entropy_{subject}_stochastic.nii.gz"))
                
                nifti_logits_foreground = nib.Nifti1Image(logits_foreground_np, affine=np.eye(4))
                nib.save(nifti_logits_foreground, os.path.join(savepath, f"logits-foreground_{subject}_stochastic.nii.gz"))
                
                nifti_logits_background = nib.Nifti1Image(logits_background_np, affine=np.eye(4))
                nib.save(nifti_logits_background, os.path.join(savepath, f"logits-background_{subject}_stochastic.nii.gz"))
                
                
            
            if basis_model_only:
                print("Yay, variance is not None")
                savepath = os.path.join(savepath_root, "basis_model_only")
                os.makedirs(savepath, exist_ok=True)
                
                mu = output.mu # logits (mu is eta_mu)
                mu_prop = torch.softmax(mu, dim=1)  # probabilities
                entropy = -torch.sum(mu_prop * torch.log(mu_prop + 1e-8), dim=1)
                
                entropy_np = entropy.detach().cpu().numpy().squeeze(0).astype(np.float32)
                prediction_np = mu.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                target_np = target.squeeze(0).detach().cpu().numpy().astype(np.uint8)
                input_np = input_volume.squeeze(0,1).detach().cpu().numpy().astype(np.float32)

                nifti_input = nib.Nifti1Image(input_np, affine=np.eye(4))
                nib.save(nifti_input, os.path.join(savepath, f"input_{subject}_basic.nii.gz"))
                
                nifti_pred = nib.Nifti1Image(prediction_np, affine=np.eye(4))
                nib.save(nifti_pred, os.path.join(savepath, f"prediction_{subject}_basic.nii.gz"))

                nifti_target = nib.Nifti1Image(target_np, affine=np.eye(4))
                nib.save(nifti_target, os.path.join(savepath, f"target_{subject}_basic.nii.gz"))

                # print("Entropy shape:", entropy.shape)
                # print("Entropy stats: min {:.4f}, max {:.4f}, mean {:.4f}".format(entropy.min().item(), entropy.max().item(), entropy.mean().item()))

                nifti_entropy = nib.Nifti1Image(entropy_np, affine=np.eye(4))
                nib.save(nifti_entropy, os.path.join(savepath, f"entropy_{subject}_basic.nii.gz"))

                
            if i == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
        
            cross_entropy = ce_forward(output.mu, target)
            
            dice = dice_score(output.mu, target)
            results['dice_score'].append(dice)
            results['cross_entropy_mean'].append(cross_entropy.item())
            performance = {key: val[-1] for key, val in results.items()}
            performance_holder.update(elem['keys'][0].item(), performance)
            eval_loader_bar.update(1)
        
        
        metrics = {key: sum(val) / len(val) for key, val in results.items()}
        if hasattr(self.model.decoder, 'cov_weighting_head'):
            metrics['basis_indices'] = {
                val: self.model.decoder.cov_weighting_head.chosen_indices.count(val) for val in np.unique(
                    self.model.decoder.cov_weighting_head.chosen_indices
                )
            }
            self.model.decoder.cov_weighting_head.track_chosen_indices(False)
        
            print(metrics['basis_indices'])
        if basis_model_only:
            self.basis_model_results_keeper = performance_holder
        else:
            self.trained_models_results_keeper = performance_holder
        self.model.basis_model_only(False)
        
        return metrics, performance_holder.mean()
        
    @torch.no_grad()
    def run_evaluation(self, eval_loader = None, basis_model_only = False):

        if eval_loader is None:
            eval_loader = self.eval_loader

        max_iter = 1000
        self.model.eval()
        num_eval_steps = len(eval_loader.generator._data.identifiers)
        eval_loader_bar = tqdm(desc = 'Running evaluation', total = len(eval_loader.generator._data.identifiers))
        performance_holder = PerformanceHolder()
        
        results = {'dice_score': [], 'cross_entropy_mean': []}
        ce_forward = nn.CrossEntropyLoss()
        if basis_model_only:
            self.model.basis_model_only()
        
        has_been_identified = {key: False for key in eval_loader.generator._data.identifiers}
        wrong_hits_counter = 0
        step = 0
        while step < max_iter:
            elem = next(eval_loader)
            if has_been_identified[elem['keys'][0].item()]:
                wrong_hits_counter += 1
                continue
        
            input_volume, target = self.handle_various_input(elem)
            input_volume = input_volume.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            output = self.model(input_volume, target, reduction = 'mean')
            
            if step == 0:
                for loss_type in output.loss_decomp.keys():
                    results[loss_type] = []
            for loss_type, val in output.loss_decomp.items():
                    results[loss_type].append(val)
            
            cross_entropy = ce_forward(output.mu, target)
            dice = dice_score(output.mu, target)
            results['dice_score'].append(dice)
            results['cross_entropy_mean'].append(cross_entropy.item())
            performance = {key: val[-1] for key, val in results.items()}
            performance_holder.update(elem['keys'][0].item(), performance)
            has_been_identified[elem['keys'][0].item()] = True

            eval_loader_bar.update(1)
            step += 1
            if all(has_been_identified.values()):
                break
        
        print("Number of wrong hits by this stupid code was:", wrong_hits_counter)
        assert len(set(performance_holder.performances.keys()) - set(has_been_identified.keys())) == 0
        
     
        metrics = {key: sum(val) / len(val) for key, val in results.items()}

        if basis_model_only:
            self.basis_model_results_keeper = performance_holder
        else:
            self.trained_models_results_keeper = performance_holder
        self.model.basis_model_only(False)
        return metrics, performance_holder.mean()


    def save_model(self, epoch):

        model_save_path = os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch))
        torch.save(self.model.state_dict(), model_save_path)
        print("Saved model to", os.path.join(self.output_dir, 'model_epoch_{}.pth'.format(epoch)))

        if self.last_saved_model_path:
            os.remove(self.last_saved_model_path)
            self.last_saved_model_path = model_save_path

    def finalize(self):

        self.train_loader.__del__()

        if not isinstance(self.eval_loader, HackyEvalLoader):
            self.eval_loader.__del__()


    def train(self, experiment_name = ""):

        logger = Logger(self.output_dir, experiment_name=experiment_name)

        performance = {'dice_score': -1}
        best_performance = 0
        self.model.to(self.device)
        basis_metrics, mean_perf = self.run_evaluation_new(basis_model_only=True)

        print(basis_metrics)
        print(mean_perf)
    
        logger.write(basis_metrics, key = 'init_metrics')
        logger.write(self.training_kwargs, 'training_kwargs')

        for epoch in range(self.training_kwargs['num_epochs']):
            self.train_one_epoch(epoch, last_performance = performance)
            performance, mean_perf = self.run_evaluation_new()
            
            basis_model_performance = self.basis_model_results_keeper.mean_with_other(
                self.trained_models_results_keeper
            )
            trained_model_performance = self.trained_models_results_keeper.mean_with_other(
                self.basis_model_results_keeper
            )

            performance_diff = trained_model_performance - basis_model_performance
            self.last_performance_diff = performance_diff.performances
            if performance['dice_score'].item() > best_performance:
                self.save_model(epoch)
                best_performance = performance['dice_score'].item()
            
            logger.write(performance_diff.performances, key = 'performance_diff')
            logger.write(performance, key = 'training_performance')

    def evaluate(self):
        # performance, mean_perf = self.run_evaluation_new_to_get_variance()
        performance, mean_perf = self.run_evaluation_new_to_get_variance(basis_model_only=self.model_kwargs['basic'])


def run_weighting_grid_search(args):
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
        'num_val_iterations': 130,
    }

    base_output_dir = '/scratch/pjtka/ndseg_output/grid_search'
    ce_values = [0.9, 1.0, 1.1]
    dice_values = [0.2, 0.5]
    kl_values = [1e-5, 1e-4, 1e-3]

    for ce in ce_values:
        for di in dice_values:
            for kl in kl_values:
                loss_kwargs = {
                        'lambda_ce':ce,
                        'lambda_dice':di,
                        'lambda_nll': 1.0,
                        'lambda_kl': kl
                    }

                training_kwargs['loss_kwargs'] = loss_kwargs
                experiment_name = 'grid_search_loss'
                outdir = os.path.join(base_output_dir, f"ce_{ce}_dice_{di}_kl_{kl}")
                training_kwargs['output_dir'] = outdir
                run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=experiment_name)


def run_experiment(model_kwargs, training_kwargs, experiment_name, num_runs = 1):
    
    for i in range(num_runs):
        trainer = Trainer(model_kwargs=model_kwargs, training_kwargs=training_kwargs)
        trainer.train(experiment_name=f"{experiment_name}_run_{i}")
        trainer.finalize()
        
        
def run_experiment_eval(model_kwargs, training_kwargs, experiment_name, num_runs = 1):
    
    for i in range(num_runs):
        trainer = Trainer(model_kwargs=model_kwargs, training_kwargs=training_kwargs)
        trainer.evaluate()
        trainer.finalize()


def run_ppt(args):
    model_kwargs = {
        'checkpoint_path': '/scratch/pjtka/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'num_samples_train': 5,
        'num_samples_inference': 30,
        'sample_type': 'torch'
    }

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        
        'eval_loader_data_path': '/scratch/pjtka/pancreas_validation',
    }

    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)
    


def run_basic(args):
    model_kwargs = {
        'checkpoint_path': '/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/home/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'num_samples_train': 5,
        'num_samples_inference': 30
    }

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },

        'eval_loader_data_path': '/home/awias/data/pancreas_validation',
    }

    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)
    
    
def run_basic_eval(args):
    model_kwargs = {
        'checkpoint_path': '/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/checkpoints/exp_basic_run_4_model_epoch_10.pth', #'/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },
        'path_to_base': '/home/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'num_samples_train': 5,
        'num_samples_inference': 30,
        'basic': False
    }
    
    # model_kwargs = {
    #     'checkpoint_path': '/home/awias/data/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
    #     'loss_kwargs': {
    #                     'lambda_ce':1.0,
    #                     'lambda_dice':1.0,
    #                     'lambda_nll': 1.0,
    #                     'lambda_kl': 1e-4
    #                 },
    #     'path_to_base': '/home/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
    #     'num_samples_train': 5,
    #     'num_samples_inference': 100,
    #     'basic': True
    # }

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 1e-4
                    },

        'eval_loader_data_path': '/home/awias/data/pancreas_validation',
        }

    run_experiment_eval(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)


def run_multiple_bases(args):

    model_kwargs = {
        'checkpoint_path': '/scratch/pjtka/nnUNet/nnUNet_results/Dataset004_TotalSegmentatorPancreas/nnUNetTrainerNoMirroring__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'loss_kwargs': {
                        'lambda_ce':1.0,
                        'lambda_dice':1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 5*1e-4
                    },
        'path_to_base': '/scratch/awias/data/nnUNet/info_dict_TotalSegmentatorPancreas.pkl',
        'model_type': 'weighted_basis',
        'cov_weighting_kwargs': {
            'num_bases': 2,
            'sample_type': 'weighted'
        },
        'num_samples_train': 5,
        'num_samples_inference': 30
    }

    training_kwargs = {
        'num_epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'output_dir': args.outdir,
        'num_iterations_per_epoch': 250,
        'num_val_iterations': 5,
        'loss_kwargs': {
                        'lambda_ce': 1.0,
                        'lambda_dice': 1.0,
                        'lambda_nll': 1.0,
                        'lambda_kl': 5*1e-4
                    },

        'eval_loader_data_path': '/scratch/pjtka/pancreas_validation',
        'gradient_accumulation': False
    }


    run_experiment(model_kwargs=model_kwargs, training_kwargs=training_kwargs, experiment_name=args.exp_name, num_runs = args.num_runs)


from argparse import ArgumentParser, Namespace
import nibabel as nib

def main():
    USE_ARGPARSE = False  # Skift til True for at bruge kommandolinjeargumenter

    if USE_ARGPARSE:
        parser = ArgumentParser()
        parser.add_argument('--exp_type', type=str, default='basic', nargs="+")
        parser.add_argument('--exp_name', type=str, default='basic', nargs="+")
        parser.add_argument('--num_runs', type=int, default=1)
        parser.add_argument('--outdir', type=str, default='/scratch/awias/data/ndseg_output', nargs="+")
        args = parser.parse_args()
    else:
        # Hardkodede argumenter
        args = Namespace(
            exp_type=['basic'],
            exp_name=['basic'],
            num_runs=1,
            outdir=['/home/awias/data/ndseg_output']
        )

    print("Kører med følgende argumenter:")
    print(args)

    exp_type_to_func = {
        'basic': run_basic_eval,
        'grid': run_weighting_grid_search,
        'multi_basis': run_multiple_bases,
        'ppt': run_ppt
    }

    assert len(args.exp_type) == len(args.exp_name), \
        'We require new name for each experiment type otherwise it will overwrite'

    if not isinstance(args.outdir, list):
        args.outdir = [args.outdir]

    if len(args.outdir) == 1:
        args.outdir = args.outdir * len(args.exp_type)

    iterator = zip(args.exp_type, args.exp_name, args.outdir)

    for exp_type, exp_name, outdir in iterator:
        run_args = Namespace(
            exp_type=exp_type,
            exp_name=exp_name,
            outdir=outdir,
            num_runs=args.num_runs
        )
        exp_type_to_func[exp_type](run_args)


if __name__ == '__main__':
    main()

        













