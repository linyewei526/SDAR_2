from contextlib import contextmanager
import torch

@contextmanager
def with_default_dtype(dtype):
    """Context manager to temporarily set torch default dtype"""
    _dtype_original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(_dtype_original)
