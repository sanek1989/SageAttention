from __future__ import annotations
from typing import ClassVar, Protocol, runtime_checkable
import torch
from .core import (
    sageattn,
    sageattn_varlen,
    sageattn_qk_int8_pv_fp16_triton,
    sageattn_qk_int8_pv_fp16_cuda,
    sageattn_qk_int8_pv_fp8_cuda,
    sageattn_qk_int8_pv_fp8_cuda_sm90
)

# Регистр поддерживаемых архитектур
_SM_ARCH_REGISTRY = {
    75: 'qk_int8_sv_f16_accum_f32_attn_sm75',
    80: 'qk_int8_sv_f16_accum_f32_attn_sm80',
    89: 'qk_int8_sv_f16_accum_f32_attn_sm89',
    90: 'qk_int8_sv_f16_accum_f32_attn_sm90'
}

def _get_gpu_compute_capability() -> tuple[int, int]:
    """Определяет точную SM-версию GPU (major, minor)"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.cuda.get_device_capability(0)

def _load_optimal_kernel() -> None:
    """Динамически загружает оптимальное ядро для текущего GPU"""
    try:
        major, minor = _get_gpu_compute_capability()
        sm_version = major * 10 + minor
        
        # Проверка минимальной версии CUDA для SM75 (T4)
        if sm_version == 75 and torch.version.cuda < "11.0":
            raise RuntimeError(f"SM75 (T4) requires CUDA 11.0+, current: {torch.version.cuda}")
        module_name = _SM_ARCH_REGISTRY.get(sm_version)
        
        if not module_name:
            raise ImportError(f"Unsupported SM version: {sm_version}")
            
        module = __import__(
            f"sageattention.csrc.qattn.{module_name}",
            fromlist=['*']
        )
        globals().update({k: getattr(module, k) for k in dir(module)})
        
    except ImportError as e:
        raise RuntimeError(
            f"Failed to load kernel for SM{sm_version}: {str(e)}"
        ) from e

# Инициализация оптимального ядра при импорте
_load_optimal_kernel()

@runtime_checkable
class KernelFeature(Protocol):
    """Протокол для проверки возможностей ядра"""
    SUPPORTED_SM: ClassVar[int]
    MIN_CUDA_VERSION: ClassVar[str]