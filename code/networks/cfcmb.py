"""
Class-Feature Contrastive Memory Bank (CFCMB) for Semi-Supervised Learning

This module implements a memory bank that maintains per-class prototype vectors
and optional queues for contrastive learning in semi-supervised 3D segmentation.

Features:
- Per-class prototype vectors updated via EMA
- Optional per-class queues for negative sampling (FIFO)
- Different momentum rates for labeled vs unlabeled samples
- Prototype-based contrastive loss (InfoNCE)
- Device-aware operations and L2 normalization
- Safe handling of empty inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ClassFeatureMemoryBank(nn.Module):
    """
    Class-Feature Contrastive Memory Bank for maintaining per-class prototypes
    and computing prototype-based contrastive losses.
    """
    
    def __init__(
        self,
        num_classes: int,
        feat_dim: int = 128,
        queue_size: int = 512,
        momentum_labeled: float = 0.9,
        momentum_unlabeled: float = 0.98,
        temperature: float = 0.15
    ):
        """
        Initialize the Class-Feature Memory Bank.
        
        Args:
            num_classes: Number of classes
            feat_dim: Feature dimension
            queue_size: Size of negative queues per class (0 to disable queues)
            momentum_labeled: EMA momentum for labeled samples (larger = faster update)
            momentum_unlabeled: EMA momentum for unlabeled samples (smaller = slower update)
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.momentum_labeled = momentum_labeled
        self.momentum_unlabeled = momentum_unlabeled
        self.temperature = temperature
        
        # Per-class prototype vectors [num_classes, feat_dim]
        self.register_buffer('prototypes', torch.randn(num_classes, feat_dim))
        self.register_buffer('prototype_initialized', torch.zeros(num_classes, dtype=torch.bool))
        
        # Optional per-class queues for negatives [num_classes, queue_size, feat_dim]
        if queue_size > 0:
            self.register_buffer('queues', torch.randn(num_classes, queue_size, feat_dim))
            self.register_buffer('queue_ptrs', torch.zeros(num_classes, dtype=torch.long))
            self.register_buffer('queue_initialized', torch.zeros(num_classes, dtype=torch.bool))
        else:
            self.queues = None
            self.queue_ptrs = None
            self.queue_initialized = None
        
        # Initialize prototypes with L2 normalization
        self._initialize_prototypes()
    
    def _initialize_prototypes(self):
        """Initialize prototypes with random L2-normalized vectors."""
        with torch.no_grad():
            self.prototypes.normal_(0, 1)
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            if self.queues is not None:
                self.queues.normal_(0, 1)
                self.queues = F.normalize(self.queues, p=2, dim=2)
    
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """L2 normalize features along the last dimension."""
        return F.normalize(features, p=2, dim=-1)
    
    def update_with_labeled(
        self, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Update prototypes using labeled features.
        
        Args:
            features: Feature vectors [N, feat_dim], must be detached and normalized
            labels: Corresponding labels [N]
            max_samples_per_class: Maximum samples to use per class (for memory efficiency)
        """
        if features.size(0) == 0:
            return
        
        features = self._normalize_features(features)
        
        with torch.no_grad():
            for class_id in range(self.num_classes):
                class_mask = (labels == class_id)
                if not class_mask.any():
                    continue
                
                class_features = features[class_mask]
                
                # Sample subset if too many features
                if max_samples_per_class is not None and class_features.size(0) > max_samples_per_class:
                    indices = torch.randperm(class_features.size(0))[:max_samples_per_class]
                    class_features = class_features[indices]
                
                # Compute mean feature for this class
                class_mean = class_features.mean(dim=0)
                class_mean = self._normalize_features(class_mean)
                
                # EMA update of prototype
                if self.prototype_initialized[class_id]:
                    self.prototypes[class_id] = (
                        self.momentum_labeled * self.prototypes[class_id] + 
                        (1 - self.momentum_labeled) * class_mean
                    )
                else:
                    self.prototypes[class_id] = class_mean
                    self.prototype_initialized[class_id] = True
                
                # Normalize updated prototype
                self.prototypes[class_id] = self._normalize_features(self.prototypes[class_id])
                
                # Update queue if enabled
                if self.queues is not None:
                    self._update_queue(class_id, class_features)
    
    def update_with_unlabeled(
        self,
        features: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence: torch.Tensor,
        conf_threshold: float = 0.9,
        max_samples_per_class: Optional[int] = None
    ):
        """
        Update prototypes using high-confidence unlabeled features.
        
        Args:
            features: Feature vectors [N, feat_dim], must be detached and normalized
            pseudo_labels: Pseudo labels [N]
            confidence: Confidence scores [N]
            conf_threshold: Minimum confidence to use for updates
            max_samples_per_class: Maximum samples to use per class
        """
        if features.size(0) == 0:
            return
        
        features = self._normalize_features(features)
        
        # Filter by confidence
        high_conf_mask = confidence >= conf_threshold
        if not high_conf_mask.any():
            return
        
        features = features[high_conf_mask]
        pseudo_labels = pseudo_labels[high_conf_mask]
        
        with torch.no_grad():
            for class_id in range(self.num_classes):
                class_mask = (pseudo_labels == class_id)
                if not class_mask.any():
                    continue
                
                class_features = features[class_mask]
                
                # Sample subset if too many features
                if max_samples_per_class is not None and class_features.size(0) > max_samples_per_class:
                    indices = torch.randperm(class_features.size(0))[:max_samples_per_class]
                    class_features = class_features[indices]
                
                # Compute mean feature for this class
                class_mean = class_features.mean(dim=0)
                class_mean = self._normalize_features(class_mean)
                
                # EMA update with smaller momentum (more conservative)
                if self.prototype_initialized[class_id]:
                    self.prototypes[class_id] = (
                        self.momentum_unlabeled * self.prototypes[class_id] + 
                        (1 - self.momentum_unlabeled) * class_mean
                    )
                else:
                    # Initialize with unlabeled if no labeled samples seen yet
                    self.prototypes[class_id] = class_mean
                    self.prototype_initialized[class_id] = True
                
                # Normalize updated prototype
                self.prototypes[class_id] = self._normalize_features(self.prototypes[class_id])
                
                # Update queue if enabled
                if self.queues is not None:
                    self._update_queue(class_id, class_features)
    
    def _update_queue(self, class_id: int, features: torch.Tensor):
        """Update the feature queue for a specific class (FIFO)."""
        if self.queues is None:
            return
        
        batch_size = features.size(0)
        ptr = self.queue_ptrs[class_id]
        
        if batch_size > self.queue_size:
            # If batch is larger than queue, just take the last queue_size samples
            features = features[-self.queue_size:]
            batch_size = self.queue_size
            ptr = 0
        
        # Update queue
        if ptr + batch_size <= self.queue_size:
            self.queues[class_id, ptr:ptr + batch_size] = features
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.queues[class_id, ptr:] = features[:remaining]
            self.queues[class_id, :batch_size - remaining] = features[remaining:]
        
        # Update pointer
        self.queue_ptrs[class_id] = (ptr + batch_size) % self.queue_size
        self.queue_initialized[class_id] = True
    
    def proto_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        temperature: Optional[float] = None,
        conf_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute prototype-based contrastive loss (InfoNCE).
        
        Args:
            features: Feature vectors [N, feat_dim]
            labels: Labels [N]
            temperature: Temperature for softmax (uses self.temperature if None)
            conf_weights: Confidence weights [N] for weighting the loss
        
        Returns:
            Contrastive loss scalar
        """
        if features.size(0) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        features = self._normalize_features(features)
        temp = temperature if temperature is not None else self.temperature
        
        # Compute similarity with all prototypes [N, num_classes]
        similarities = torch.mm(features, self.prototypes.t()) / temp
        
        # Create loss for each sample
        losses = F.cross_entropy(similarities, labels.long(), reduction='none')
        
        # Apply confidence weighting if provided
        if conf_weights is not None:
            losses = losses * conf_weights
        
        return losses.mean()
    
    def get_prototype_norms(self) -> torch.Tensor:
        """Get L2 norms of all prototypes for logging."""
        return torch.norm(self.prototypes, p=2, dim=1)
    
    def get_prototype_similarities(self) -> torch.Tensor:
        """Get pairwise cosine similarities between prototypes."""
        normalized_protos = self._normalize_features(self.prototypes)
        return torch.mm(normalized_protos, normalized_protos.t())
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass - compute contrastive loss."""
        return self.proto_contrastive_loss(features, labels, **kwargs)


def sample_features_per_class(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_samples_per_class: int,
    spatial_dims: Tuple[int, ...] = (2, 3, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample features per class from spatial feature maps.
    
    Args:
        features: Feature maps [B, C, D, H, W]
        labels: Label maps [B, 1, D, H, W] or [B, D, H, W]
        max_samples_per_class: Maximum samples per class per batch
        spatial_dims: Spatial dimensions to flatten
    
    Returns:
        Tuple of (sampled_features [N, C], sampled_labels [N])
    """
    device = features.device
    
    # Ensure labels have correct shape
    if labels.dim() == 4:  # [B, D, H, W]
        labels = labels.unsqueeze(1)  # [B, 1, D, H, W]
    
    # Flatten spatial dimensions
    B, C = features.shape[:2]
    features_flat = features.flatten(2)  # [B, C, D*H*W]
    labels_flat = labels.flatten(2)      # [B, 1, D*H*W]
    
    all_features = []
    all_labels = []
    
    for b in range(B):
        batch_features = features_flat[b].transpose(0, 1)  # [D*H*W, C]
        batch_labels = labels_flat[b, 0]                   # [D*H*W]
        
        # Get unique classes in this batch
        unique_classes = torch.unique(batch_labels)
        
        for class_id in unique_classes:
            class_mask = (batch_labels == class_id)
            class_features = batch_features[class_mask]
            
            if class_features.size(0) == 0:
                continue
            
            # Sample up to max_samples_per_class
            if class_features.size(0) > max_samples_per_class:
                indices = torch.randperm(class_features.size(0), device=device)[:max_samples_per_class]
                class_features = class_features[indices]
            
            # Add to lists
            all_features.append(class_features)
            all_labels.append(torch.full((class_features.size(0),), class_id, 
                                       device=device, dtype=batch_labels.dtype))
    
    if len(all_features) == 0:
        # Return empty tensors if no features found
        return (torch.empty(0, C, device=device), 
                torch.empty(0, device=device, dtype=labels.dtype))
    
    # Concatenate all features and labels
    sampled_features = torch.cat(all_features, dim=0)
    sampled_labels = torch.cat(all_labels, dim=0)
    
    return sampled_features, sampled_labels