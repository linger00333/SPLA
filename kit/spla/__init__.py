"""SPLA kit: sequence-prior local alignment for unregistered RGB-IR UAV video.

NumPy parts (geometry, prior_estimator) have no deep-learning dependency.
The PyTorch parts live in spla.modules and spla.ultralytics_hooks.
"""
from .geometry import (MODELS, apply_theta, canvas_matrix, choose_model, compose_augmentation,
                       fit_transform, offset_field, params_to_theta, set_zoom_aspect, theta_on_canvas,
                       theta_to_params)
from .prior_estimator import OnlinePriorEstimator, PriorConfig, dataset_prior

__all__ = [
    "MODELS", "apply_theta", "canvas_matrix", "choose_model", "compose_augmentation",
    "fit_transform", "offset_field", "params_to_theta", "set_zoom_aspect", "theta_on_canvas", "theta_to_params",
    "OnlinePriorEstimator", "PriorConfig", "dataset_prior",
]
