# Weak labeling module - DEPRECATED
# This module has been replaced by LLM judges and transformer cross-encoders
# Kept as stub for backward compatibility

import warnings

def deprecated_weak_labeling(*args, **kwargs):
    warnings.warn(
        "Weak labeling has been deprecated. Use LLM judges and cross-encoders instead.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Weak labeling has been removed. Use the new pipeline.")