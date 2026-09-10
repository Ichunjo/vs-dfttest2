import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Final

__all__ = ["get_location", "get_sigma", "get_window", "get_window_value", "normalize"]

COSINE_WINDOWS: Final[Mapping[int, tuple[float, ...]]] = {
    0: (0.5, -0.5),  # hanning
    1: (0.53836, -0.46164),  # hamming
    2: (0.42, -0.5, 0.08),  # blackman
    3: (0.35875, -0.48829, 0.14128, -0.01168),  # 4 term blackman-harris
    5: (
        0.27105140069342415,
        -0.433297939234486060,
        0.218122999543110620,
        -0.065925446388030898,
        0.010811742098372268,
        -7.7658482522509342e-4,
        1.3887217350903198e-5,
    ),  # 7 term blackman-harris
    6: (0.2810639, -0.5208972, 0.1980399),  # flat top
    10: (0.355768, -0.487396, 0.144232, -0.012604),  # nuttall
    11: (0.3635819, -0.4891775, 0.1365995, -0.0106411),  # blackman-nuttall
}


def _i0(p: float) -> float:
    p /= 2
    n = t = d = 1.0
    k = 1
    while True:
        n *= p
        d *= k
        v = n / d
        t += v * v
        k += 1
        if k >= 15 or v <= 1e-8:
            break
    return t


# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/
# bc5e0186a7f309556f20a8e9502f2238e39179b8/DFTTest/DFTTest.cpp#L518
def normalize(window: Sequence[float], size: int, step: int) -> list[float]:
    sq_sums = [math.sqrt(sum(window[h] ** 2 for h in range(rem, size, step))) for rem in range(step)]
    return [w / sq_sums[q % step] for q, w in enumerate(window)]


# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/
# bc5e0186a7f309556f20a8e9502f2238e39179b8/DFTTest/DFTTest.cpp#L462
def get_window_value(location: float, size: int, mode: int, beta: float) -> float:
    temp = math.pi * location / size

    if (coeffs := COSINE_WINDOWS.get(mode)) is not None:
        return sum(c * math.cos(2 * k * temp) for k, c in enumerate(coeffs))

    match mode:
        case 4:  # kaiser-bessel
            v = 2 * location / size - 1
            return _i0(math.pi * beta * math.sqrt(1 - v * v)) / _i0(math.pi * beta)
        case 7:  # rectangular
            return 1.0
        case 8:  # Bartlett
            return 1.0 - 2.0 * abs(location - size / 2) / size
        case 9:  # bartlett-hann
            return 0.62 - 0.48 * (location / size - 0.5) - 0.38 * math.cos(2 * temp)
        case _:
            raise ValueError(f"unknown window: {mode}")


# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/
# bc5e0186a7f309556f20a8e9502f2238e39179b8/DFTTest/DFTTest.cpp#L461
def get_window(
    radius: int,
    block_size: int,
    block_step: int,
    spatial_window_mode: int,
    spatial_beta: float,
    temporal_window_mode: int,
    temporal_beta: float,
) -> list[float]:
    temporal_window = [
        get_window_value(i + 0.5, 2 * radius + 1, temporal_window_mode, temporal_beta) for i in range(2 * radius + 1)
    ]
    spatial_window = normalize(
        [get_window_value(i + 0.5, block_size, spatial_window_mode, spatial_beta) for i in range(block_size)],
        block_size,
        block_step,
    )

    scale = 1.0 / (math.sqrt(2 * radius + 1) * block_size)
    spatial_2d = [s1 * s2 * scale for s1, s2 in itertools.product(spatial_window, spatial_window)]
    return [t * s for t, s in itertools.product(temporal_window, spatial_2d)]


# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/
# bc5e0186a7f309556f20a8e9502f2238e39179b8/DFTTest/DFTTest.cpp#L581
def get_location(position: float, length: int) -> float:
    if length == 1:
        return 0.0
    half = length // 2
    return (length - position if position > half else position) / half


# https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/
# bc5e0186a7f309556f20a8e9502f2238e39179b8/DFTTest/DFTTest.cpp#L581
def get_sigma(position: float, length: int, func: Callable[[float], float]) -> float:
    return 1.0 if length == 1 else func(get_location(position, length))
