from dataclasses import dataclass

from vapoursynth import core

type BackendInstance = Backend.cuFFT | Backend.NVRTC | Backend.CPU | Backend.GCC | Backend.hipFFT | Backend.HIPRTC
type BackendT = BackendInstance | type[BackendInstance]
backendT = BackendT


class Backend:
    @dataclass(frozen=False)
    class cuFFT:
        device_id: int = 0
        in_place: bool = True

    @dataclass(frozen=False)
    class NVRTC:
        device_id: int = 0
        num_streams: int = 1

    @dataclass(frozen=False)
    class CPU:
        opt: int = 0

    @dataclass(frozen=False)
    class GCC: ...

    @dataclass(frozen=False)
    class hipFFT:
        device_id: int = 0
        in_place: bool = True

    @dataclass(frozen=False)
    class HIPRTC:
        device_id: int = 0
        num_streams: int = 1


def select_backend(backend: BackendT | None, sbsize: int, tbsize: int) -> BackendT:
    if backend is not None:
        return backend

    candidates: tuple[tuple[str, type[BackendT]], ...]
    if sbsize == 16 and tbsize in (1, 3, 5, 7):
        candidates = (
            ("dfttest2_nvrtc", Backend.NVRTC),
            ("dfttest2_hiprtc", Backend.HIPRTC),
            ("dfttest2_cuda", Backend.cuFFT),
            ("dfttest2_hip", Backend.hipFFT),
            ("dfttest2_cpu", Backend.CPU),
            ("dfttest2_gcc", Backend.GCC),
        )
    else:
        candidates = (
            ("dfttest2_cuda", Backend.cuFFT),
            ("dfttest2_hip", Backend.hipFFT),
        )

    for plugin_name, backend_cls in candidates:
        if hasattr(core, plugin_name):
            return backend_cls()

    raise RuntimeError("No backend are available")
