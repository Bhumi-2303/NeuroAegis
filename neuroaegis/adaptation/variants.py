import torch
import torch.nn as nn
from typing import Any
import torch.optim as optim
from neuroaegis.models.schemas import ModelOutput

class VariantA_DirectTransfer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        return self.model(x)

class VariantB_Normalization(nn.Module):
    def __init__(self, model: nn.Module, calibration_loader: Any):
        super().__init__()
        self.model = model
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self._fit_statistics(calibration_loader)
        
    def _fit_statistics(self, loader: Any):
        # Compute mean and std over the calibration subset
        sum_x = 0.0
        sum_sq_x = 0.0
        n_elements = 0
        
        for x_batch, _ in loader:
            sum_x += x_batch.sum(dim=(0, 2), keepdim=True)
            sum_sq_x += (x_batch ** 2).sum(dim=(0, 2), keepdim=True)
            n_elements += x_batch.size(0) * x_batch.size(2)
            
        mean = sum_x / n_elements
        variance = (sum_sq_x / n_elements) - (mean ** 2)
        std = torch.sqrt(variance + 1e-8)
        
        self.mean = mean
        self.std = std
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        x_norm = (x - self.mean) / self.std
        return self.model(x_norm)

class VariantC_Calibration(nn.Module):
    def __init__(self, model: nn.Module, calibration_loader: Any):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._fit_temperature(calibration_loader)
        
    def _fit_temperature(self, loader: Any):
        # Temperature scaling via LBFGS on calibration logits
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                out = self.model(x_batch)
                logits_list.append(out.logits)
                labels_list.append(y_batch)
                
        logits = torch.cat(logits_list).detach()
        labels = torch.cat(labels_list).float().detach()
        
        # Optimize temperature
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            # Binary Cross Entropy
            loss = nn.BCEWithLogitsLoss()(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.model(x)
        out.logits = out.logits / self.temperature
        return out

class VariantD_Combined(nn.Module):
    """
    Harmonization + calibration combined (B + C).
    """
    def __init__(self, model: nn.Module, calibration_loader: Any):
        super().__init__()
        # 1. First fit normalization using a dummy wrapper to collect stats
        self.normalized_model = VariantB_Normalization(model, calibration_loader)
        
        # 2. Then fit temperature using the normalized inputs
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._fit_temperature(calibration_loader)
        
    def _fit_temperature(self, loader: Any):
        self.normalized_model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                out = self.normalized_model(x_batch)
                logits_list.append(out.logits)
                labels_list.append(y_batch)
                
        logits = torch.cat(logits_list).detach()
        labels = torch.cat(labels_list).float().detach()
        
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = nn.BCEWithLogitsLoss()(scaled_logits, labels)
            loss.backward()
            return loss
            
        optimizer.step(eval_loss)
        
    def forward(self, x: torch.Tensor) -> ModelOutput:
        out = self.normalized_model(x)
        out.logits = out.logits / self.temperature
        return out
