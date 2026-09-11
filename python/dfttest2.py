import itertools
import math
import sys
from collections.abc import Callable, Sequence
from string import Template
from typing import Any, Literal

from vapoursynth import SampleType, VideoNode, core

if sys.version_info >= (3, 13):
    from typing import TypeIs
else:
    TypeIs = Any

from .backends import Backend, BackendT, select_backend
from .helpers import get_location, get_sigma, get_window

type FREQ = float
type SIGMA = float
type LocationData = Sequence[tuple[FREQ, SIGMA]] | Sequence[float]

__all__ = ["DFTTest", "DFTTest2"]


def DFTTest(
    clip: VideoNode,
    ftype: Literal[0, 1, 2, 3, 4] = 0,
    sigma: float = 8.0,
    sigma2: float = 8.0,
    pmin: float = 0.0,
    pmax: float = 500.0,
    sbsize: int = 16,
    smode: Literal[0, 1] = 1,
    sosize: int = 12,
    tbsize: int = 3,
    # tmode=0, tosize=0
    swin: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 0,
    twin: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 7,
    sbeta: float = 2.5,
    tbeta: float = 2.5,
    zmean: bool = True,
    f0beta: float = 1.0,
    nlocation: Sequence[int] | None = None,
    alpha: float | None = None,
    slocation: LocationData | None = None,
    ssx: LocationData | None = None,
    ssy: LocationData | None = None,
    sst: LocationData | None = None,
    ssystem: Literal[0, 1] = 0,
    planes: int | Sequence[int] | None = None,
    backend: BackendT | None = None,
) -> VideoNode:
    """2D/3D frequency domain denoiser

    The interface is compatible with core.dfttest.DFTTest by HolyWu.

    Args:
        clip: Clip to process.

            Any format with either integer sample type of 8-16 bit depth
            or float sample type of 32 bit depth is supported.

        ftype: Controls the filter type.

            Possible settings are:
                0: generalized wiener filter
                    mult = max((psd - sigma) / psd, 0) ^ f0beta

                1: hard threshold
                    mult = psd < sigma ? 0.0 : 1.0

                2: multiplier
                    mult = sigma

                3: multiplier switched based on psd value
                    mult = (psd >= pmin && psd <= pmax) ? sigma : sigma2

                4: multiplier modified based on psd value and range
                    mult = sigma * sqrt((psd * pmax) / ((psd + pmin) * (psd + pmax)))

            The real and imaginary parts of each complex dft coefficient are multiplied
            by the corresponding 'mult' value.

            ** psd = magnitude squared = real*real + imag*imag

        sigma, sigma2: Value of sigma and sigma2.
            If using the slocation parameter then the sigma parameter is ignored.

        pmin, pmax: Used as described in the ftype parameter description.

        sbsize: Sets the length of the sides of the spatial window.
            Must be 1 or greater. Must be odd if using smode=0.

        smode: Sets the mode for spatial operation.
            Currently only tmode=1 is implemented.

        sosize: Sets the spatial overlap amount.
            Must be in the range 0 to sbsize-1 (inclusive).
            If sosize is greater than sbsize>>1, then sbsize%(sbsize-sosize) must equal 0.
            In other words, overlap greater than 50% requires that sbsize-sosize be a divisor of sbsize.

        tbsize: Sets the length of the temporal dimension (i.e. number of frames).
            Must be at least 1. Must be odd if using tmode=0.

        tmode: Sets the mode for temporal operation.
            Currently only tmode=0 is implemented.

        tosize: Sets the temporal overlap amount.
            Must be in the range 0 to tbsize-1 (inclusive).
            If tosize is greater than tbsize>>1, then tbsize%(tbsize-tosize) must equal 0.
            In other words, overlap greater than 50% requires that tbsize-tosize be a divisor of tbsize.

        swin, twin: Sets the type of analysis/synthesis window to be used for spatial (swin) and
            temporal (twin) processing. Possible settings:

            0: hanning
            1: hamming
            2: blackman
            3: 4 term blackman-harris
            4: kaiser-bessel
            5: 7 term blackman-harris
            6: flat top
            7: rectangular
            8: Bartlett
            9: Bartlett-Hann
            10: Nuttall
            11: Blackman-Nuttall

        sbeta,tbeta: Sets the beta value for kaiser-bessel window type.
            sbeta goes with swin, tbeta goes with twin.
            Not used unless the corresponding window value is set to 4.

        zmean: Controls whether the window mean is subtracted out (zero'd)
            prior to filtering in the frequency domain.

        f0beta: Power term in ftype=0.

        nlocation: Currently not implemented.

        slocation/ssx/ssy/sst: Used to specify functions of sigma based on frequency.
            Check the original documentation for details.

            Note that in current implementation,
            "slocation = [(0.0, 1.0), (1.0, 10.0)]"
            is equivalent to
            "slocation = [0.0, 1.0, 1.0, 10.0]"

        ssystem: Method of sigma computation.
            Check the original documentation for details.

        planes: Sets which planes will be processed.
            Any unprocessed planes will be simply copied.

        backend: Backend implementation to use.
            All available backends can be found in the dfttest2.Backend "namespace":
                dfttest2.Backend.{CPU, cuFFT, NVRTC, GCC, hipFFT, HIPRTC}

            The CPU, NVRTC and GCC backends require sbsize=16.
            The cuFFT and NVRTC backends require a CUDA-enabled system.
            The hipFFT and HIPRTC backends require a CUDA-enabled system.

            Speed: NVRTC == HIPRTC >> cuFFT > hipFFT > CPU == GCC
    """

    if (
        clip.width == 0
        or clip.height == 0
        or not clip.format
        or (clip.format.sample_type == SampleType.INTEGER and clip.format.bits_per_sample > 16)
        or (clip.format.sample_type == SampleType.FLOAT and clip.format.bits_per_sample != 32)
    ):
        raise ValueError("only constant format 8-16 bit integer and 32 bit float input supported")

    if ftype < 0 or ftype > 4:
        raise ValueError("ftype must be 0, 1, 2, 3, or 4")

    if sbsize < 1:
        raise ValueError("sbsize must be greater than or equal to 1")

    if smode != 1:
        raise ValueError('"smode" must be 1')

    if sosize > sbsize // 2 and (sbsize % (sbsize - sosize) != 0):
        raise ValueError("spatial overlap greater than 50% requires that sbsize-sosize is a divisor of sbsize")

    if tbsize < 1:
        raise ValueError('"tbsize" must be at least 1')

    if not 0 <= swin <= 11:
        raise ValueError("swin must be between 0 and 11 (inclusive)")

    if not 0 <= twin <= 11:
        raise ValueError("twin must be between 0 and 11 (inclusive)")

    if nlocation is not None:
        raise ValueError('"nlocation" must be None')

    for name, loc in (("slocation", slocation), ("ssx", ssx), ("ssy", ssy), ("sst", sst)):
        if loc and len(loc) % 2 != 0:
            raise ValueError(f"number of elements in {name} must be a multiple of 2")

    if ssystem not in (0, 1):
        raise ValueError("ssystem must be 0 or 1")

    def norm(x: float) -> float:
        if slocation is not None and ssystem == 1:
            return x
        if tbsize == 1:
            return math.sqrt(x)
        return math.cbrt(x)

    if slocation is not None:
        sigma_norm = [to_func(flatten(slocation), norm, sigma)] * 3
    elif any(ss is not None for ss in (ssx, ssy, sst)):
        sigma_norm = [to_func(flatten(ss), norm, sigma) for ss in (ssx, ssy, sst)]
    else:
        sigma_norm = sigma

    return DFTTest2(
        clip=clip,
        ftype=ftype,
        sigma=sigma_norm,
        sigma2=sigma2,
        pmin=pmin,
        pmax=pmax,
        sbsize=sbsize,
        sosize=sosize,
        tbsize=tbsize,
        swin=swin,
        twin=twin,
        sbeta=sbeta,
        tbeta=tbeta,
        zmean=zmean,
        f0beta=f0beta,
        ssystem=ssystem,
        planes=planes,
        backend=select_backend(backend, sbsize, tbsize),
    )


def DFTTest2(
    clip: VideoNode,
    ftype: Literal[0, 1, 2, 3, 4] = 0,
    sigma: float | Callable[[float], float] | Sequence[Callable[[float], float]] = 8.0,
    sigma2: float = 8.0,
    pmin: float = 0.0,
    pmax: float = 500.0,
    sbsize: int = 16,
    sosize: int = 12,
    tbsize: int = 3,
    swin: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 0,
    twin: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 7,
    sbeta: float = 2.5,
    tbeta: float = 2.5,
    zmean: bool = True,
    f0beta: float = 1.0,
    ssystem: Literal[0, 1] = 0,
    planes: int | Sequence[int] | None = None,
    backend: BackendT = Backend.cuFFT,
) -> VideoNode:
    """this interface is not stable"""

    # translate parameters
    if ftype == 0:
        if abs(f0beta - 1) < 0.00005:
            filter_type = 0
        elif abs(f0beta - 0.5) < 0.0005:
            filter_type = 6
        else:
            filter_type = 5
    else:
        filter_type = ftype

    radius = (tbsize - 1) // 2
    block_size = sbsize
    block_step = sbsize - sosize
    spatial_window_mode = swin
    temporal_window_mode = twin
    spatial_beta = sbeta
    temporal_beta = tbeta
    zero_mean = zmean
    backend_inst = backend() if isinstance(backend, type) else backend

    if isinstance(backend_inst, (Backend.CPU, Backend.NVRTC, Backend.GCC, Backend.HIPRTC)):
        if radius not in range(4):
            raise ValueError("invalid radius (tbsize)")
        if block_size != 16:
            raise ValueError("invalid block_size (sbsize)")

    t_len = 2 * radius + 1
    y_len = block_size
    x_len = block_size // 2 + 1

    if not isinstance(sigma, Sequence) and not callable(sigma):
        sigma_norm = float(sigma)
    else:
        sigma_funcs = [sigma] if callable(sigma) else list(sigma)[:3]
        sigma_funcs += [sigma_funcs[-1]] * (3 - len(sigma_funcs))
        sigma_func_x, sigma_func_y, sigma_func_t = sigma_funcs

        if ssystem == 0:
            sigmas_t = [get_sigma(t, t_len, sigma_func_t) for t in range(t_len)]
            sigmas_y = [get_sigma(y, y_len, sigma_func_y) for y in range(y_len)]
            sigmas_x = [get_sigma(x, x_len, sigma_func_x) for x in range(x_len)]

            sigma_norm = [st * sy * sx for st, sy, sx in itertools.product(sigmas_t, sigmas_y, sigmas_x)]
        else:
            locs_t_sq = [get_location(t, t_len) ** 2 for t in range(t_len)]
            locs_y_sq = [get_location(y, y_len) ** 2 for y in range(y_len)]
            locs_x_sq = [get_location(x, x_len) ** 2 for x in range(x_len)]

            inv_ndim = 1.0 / (3 if radius > 0 else 2)
            sigma_norm = [
                sigma_func_t(math.sqrt((lt + ly + lx) * inv_ndim))
                for lt, ly, lx in itertools.product(locs_t_sq, locs_y_sq, locs_x_sq)
            ]

    window = get_window(
        radius=radius,
        block_size=block_size,
        block_step=block_step,
        spatial_window_mode=spatial_window_mode,
        temporal_window_mode=temporal_window_mode,
        spatial_beta=spatial_beta,
        temporal_beta=temporal_beta,
    )

    wscale = math.fsum(w * w for w in window)

    if ftype < 2:
        if not isinstance(sigma_norm, list):
            sigma_norm *= wscale
        else:
            sigma_norm = [s * wscale for s in sigma_norm]
        sigma2 *= wscale

    if filter_type == 5:
        pmin = f0beta  # unscaled
    else:
        pmin *= wscale

    pmax *= wscale

    match backend_inst:
        case Backend.cuFFT():
            plugin = core.dfttest2_cuda
        case Backend.NVRTC():
            plugin = core.dfttest2_nvrtc
        case Backend.CPU():
            plugin = core.dfttest2_cpu
        case Backend.GCC():
            plugin = core.dfttest2_gcc
        case Backend.hipFFT():
            plugin = core.dfttest2_hip
        case Backend.HIPRTC():
            plugin = core.dfttest2_hiprtc
        case _:
            raise TypeError(f"Unknown backend: {backend_inst}")

    shape = (block_size, block_size) if radius == 0 else (t_len, block_size, block_size)
    window_freq = plugin.RDFT(data=[w * 255 for w in window], shape=shape)

    if isinstance(backend_inst, (Backend.CPU, Backend.GCC)):
        kwargs = {"opt": backend_inst.opt} if isinstance(backend_inst, Backend.CPU) else {}
        sigma_val = [sigma_norm] * (t_len * y_len * x_len) if not isinstance(sigma_norm, list) else sigma_norm
        return plugin.DFTTest(
            clip,
            window=window,
            sigma=sigma_val,
            sigma2=sigma2,
            pmin=pmin,
            pmax=pmax,
            radius=radius,
            block_size=block_size,
            block_step=block_step,
            planes=planes,
            filter_type=filter_type,
            window_freq=window_freq,
            **kwargs,
        )

    if isinstance(backend_inst, Backend.NVRTC):
        sigma_val = [sigma_norm] if not isinstance(sigma_norm, list) else sigma_norm
        return plugin.DFTTest(
            clip,
            window=window,
            sigma=sigma_val,
            sigma2=sigma2,
            pmin=pmin,
            pmax=pmax,
            filter_type=filter_type,
            radius=radius,
            block_size=block_size,
            block_step=block_step,
            zero_mean=int(zero_mean),
            window_freq=window_freq if zero_mean else None,
            planes=planes,
            device_id=backend_inst.device_id,
            num_streams=backend_inst.num_streams,
        )

    to_single = plugin.ToSingle
    sigma_str = (
        to_single(sigma_norm) if not isinstance(sigma_norm, list) else ",".join(str(to_single(x)) for x in sigma_norm)
    )

    kernel = Template(
        """
    #define FILTER_TYPE ${filter_type}
    #define ZERO_MEAN ${zero_mean}
    #define SIGMA_IS_SCALAR ${sigma_is_scalar}

    #if ZERO_MEAN
    __device__ static const float window_freq[] { ${window_freq} };
    #endif // ZERO_MEAN

    __device__ static const float window[] { ${window} };

    __device__
    static void filter(float2 & value, int x, int y, int t) {
    #if SIGMA_IS_SCALAR
        float sigma = static_cast<float>(${sigma});
    #else // SIGMA_IS_SCALAR
        __device__ static const float sigma_array[] { ${sigma} };
        float sigma = sigma_array[(t * BLOCK_SIZE + y) * (BLOCK_SIZE / 2 + 1) + x];
    #endif // SIGMA_IS_SCALAR
        [[maybe_unused]] float sigma2 = static_cast<float>(${sigma2});
        [[maybe_unused]] float pmin = static_cast<float>(${pmin});
        [[maybe_unused]] float pmax = static_cast<float>(${pmax});
        [[maybe_unused]] float multiplier {};

    #if FILTER_TYPE == 2
        value.x *= sigma;
        value.y *= sigma;
        return ;
    #endif

        float psd = value.x * value.x + value.y * value.y;

    #if FILTER_TYPE == 1
        if (psd < sigma) {
            value.x = 0.0f;
            value.y = 0.0f;
        }
        return ;
    #elif FILTER_TYPE == 0
        multiplier = fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f);
    #elif FILTER_TYPE == 3
        if (psd >= pmin && psd <= pmax) {
            multiplier = sigma;
        } else {
            multiplier = sigma2;
        }
    #elif FILTER_TYPE == 4
        multiplier = sigma * sqrtf(psd * (pmax / ((psd + pmin) * (psd + pmax) + 1e-15f)));
    #elif FILTER_TYPE == 5
        multiplier = powf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f), pmin);
    #else
        multiplier = sqrtf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f));
    #endif

        value.x *= multiplier;
        value.y *= multiplier;
    }
    """
    ).substitute(
        sigma_is_scalar=int(not isinstance(sigma_norm, list)),
        sigma=sigma_str,
        sigma2=to_single(sigma2),
        pmin=to_single(pmin),
        pmax=to_single(pmax),
        filter_type=int(filter_type),
        window_freq=",".join(str(to_single(x)) for x in window_freq),
        zero_mean=int(zero_mean),
        window=",".join(str(to_single(x)) for x in window),
    )

    gpu_kwargs = (
        {"in_place": backend_inst.in_place, "device_id": backend_inst.device_id}
        if isinstance(backend_inst, (Backend.cuFFT, Backend.hipFFT))
        else {"in_place": False, "device_id": backend_inst.device_id, "num_streams": backend_inst.num_streams}
    )

    return plugin.DFTTest(
        clip,
        kernel=kernel,
        radius=radius,
        block_size=block_size,
        block_step=block_step,
        planes=planes,
        **gpu_kwargs,
    )


def flatten(data: LocationData | None) -> list[float] | None:
    def _is_flattened_location(data: LocationData) -> TypeIs[Sequence[float]]:
        return isinstance(data[0], (int, float))

    if not data:
        return None

    if _is_flattened_location(data):
        return list(data)

    return list(itertools.chain.from_iterable(data))


def to_func(data: Sequence[float] | None, norm: Callable[[float], float], sigma: float) -> Callable[[float], float]:
    if data is None:
        return lambda _: norm(sigma)

    locations = data[::2]
    sigmas = data[1::2]
    packs = sorted(zip(locations, sigmas), key=lambda group: group[0])

    def func(x: float) -> float:
        for (x0, y0), (x1, y1) in itertools.pairwise(packs):
            if x <= x1:
                weight = (x - x0) / (x1 - x0)
                return (1 - weight) * norm(y0) + weight * norm(y1)
        raise ValueError(f"Value {x} out of range")

    return func
