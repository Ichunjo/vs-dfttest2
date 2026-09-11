#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

#include "dft_codelets.cuh"

__device__ __forceinline__ static int calc_pad_size(int size, int block_size, int block_step) {
    return size + ((size % block_size) ? block_size - size % block_size : 0) +
           max(block_size - block_step, block_step) * 2;
}

__device__ __forceinline__ static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

template <typename TYPE> __device__ __forceinline__ static float to_float(TYPE x, float scale) {
    return static_cast<float>(x) * scale;
}

template <typename TYPE> __device__ __forceinline__ static TYPE from_float(float x, float scale, float peak) {
    if constexpr (std::is_same_v<TYPE, float>) {
        return x / scale;
    } else {
        x /= scale;
        x = fmaxf(0.0f, fminf(x + 0.5f, peak));
        return static_cast<TYPE>(__float2int_rz(x));
    }
}

__device__ __forceinline__ static void apply_filter(
    float2& value,
    int x,
    int y,
    int t,
    const float* __restrict__ sigma_array,
    float sigma_scalar,
    int sigma_is_scalar,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type,
    int block_size
) {
    float sigma = sigma_is_scalar ? sigma_scalar : sigma_array[(t * block_size + y) * (block_size / 2 + 1) + x];

    if (filter_type == 2) {
        value.x *= sigma;
        value.y *= sigma;
        return;
    }

    float psd = value.x * value.x + value.y * value.y;
    if (filter_type == 1) {
        if (psd < sigma) {
            value.x = 0.0f;
            value.y = 0.0f;
        }
        return;
    }

    float multiplier;
    switch (filter_type) {
    case 0:
        multiplier = fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f);
        break;
    case 3:
        multiplier = (psd >= pmin && psd <= pmax) ? sigma : sigma2;
        break;
    case 4:
        multiplier = sigma * sqrtf(psd * (pmax / ((psd + pmin) * (psd + pmax) + 1e-15f)));
        break;
    case 5:
        multiplier = powf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f), pmin);
        break;
    case 6:
    default:
        multiplier = sqrtf(fmaxf((psd - sigma) / (psd + 1e-15f), 0.0f));
        break;
    }

    value.x *= multiplier;
    value.y *= multiplier;
}

template <int RADIUS, typename TYPE, bool ZERO_MEAN>
__device__ __forceinline__ void fused_kernel(
    float* __restrict__ dstp,
    const TYPE* __restrict__ srcp,
    const float* __restrict__ window,
    const float* __restrict__ window_freq,
    const float* __restrict__ sigma_array,
    float sigma_scalar,
    int sigma_is_scalar,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type,
    float scale,
    int block_step,
    int width,
    int height
) {
    constexpr int radius = RADIUS;
    constexpr int block_size = 16;

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    constexpr int warp_size = 32;
    constexpr int warps_per_block = 1;
    constexpr int transpose_stride = block_size + 1; // 17
    __shared__ float2 shared_transpose_buffer[warps_per_block * block_size * transpose_stride];

    int lane_id = threadIdx.x % warp_size;
    auto transpose_buffer = &shared_transpose_buffer[0];

    for (int block_id = blockIdx.x; block_id < num_blocks; block_id += gridDim.x) {
        int ix = block_id % horizontal_num;
        int iy = block_id / horizontal_num;

        if (lane_id < block_size) {
            constexpr int active_mask = 0x0000FFFF;
            float2 thread_data[(2 * radius + 1) * block_size];

// im2col
#pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto src = &srcp[(i * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
                auto local_thread_data = &thread_data[i * block_size];
#pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float*)local_thread_data)[j] = to_float(src[j * horizontal_size + lane_id], scale) *
                                                     window[(i * block_size + j) * block_size + lane_id];
                }
            }

// rdft
#pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto local_thread_data = &thread_data[i * block_size];

                __syncwarp(active_mask);
#pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float*)transpose_buffer)[j * transpose_stride + lane_id] = ((float*)local_thread_data)[j];
                }

                __syncwarp(active_mask);
#pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float*)local_thread_data)[j] = ((float*)transpose_buffer)[lane_id * transpose_stride + j];
                }

                __syncwarp(active_mask);
                rdft<block_size>((float*)local_thread_data);

#pragma unroll
                for (int j = 0; j < block_size / 2 + 1; j++) {
                    transpose_buffer[lane_id * transpose_stride + j] = local_thread_data[j];
                }

                __syncwarp(active_mask);
                if (lane_id < block_size / 2 + 1) {
#pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        local_thread_data[j] = transpose_buffer[j * transpose_stride + lane_id];
                    }

                    __syncwarp(0x000001FF);
                    dft<block_size>((float*)local_thread_data);
                }
            }

            if (lane_id < block_size / 2 + 1) {
#pragma unroll
                for (int i = 0; i < block_size; i++) {
                    dft<2 * radius + 1>((float*)&thread_data[i], block_size);
                }
            }

            // frequency_filtering
            if (lane_id < block_size / 2 + 1) {
                if constexpr (ZERO_MEAN) {
                    float gf;
                    if (lane_id == 0) {
                        gf = thread_data[0].x / window_freq[0];
                    }
                    gf = __shfl_sync(0x000001FF, gf, 0);

#pragma unroll
                    for (int i = 0; i < 2 * radius + 1; i++) {
#pragma unroll
                        for (int j = 0; j < block_size; j++) {
                            float2 local_data = thread_data[i * block_size + j];
                            float val1 = gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2];
                            float val2 =
                                gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2 + 1];
                            local_data.x -= val1;
                            local_data.y -= val2;

                            apply_filter(
                                local_data,
                                lane_id,
                                j,
                                i,
                                sigma_array,
                                sigma_scalar,
                                sigma_is_scalar,
                                sigma2,
                                pmin,
                                pmax,
                                filter_type,
                                block_size
                            );

                            local_data.x += val1;
                            local_data.y += val2;
                            thread_data[i * block_size + j] = local_data;
                        }
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < 2 * radius + 1; i++) {
#pragma unroll
                        for (int j = 0; j < block_size; j++) {
                            float2 local_data = thread_data[i * block_size + j];
                            apply_filter(
                                local_data,
                                lane_id,
                                j,
                                i,
                                sigma_array,
                                sigma_scalar,
                                sigma_is_scalar,
                                sigma2,
                                pmin,
                                pmax,
                                filter_type,
                                block_size
                            );
                            thread_data[i * block_size + j] = local_data;
                        }
                    }
                }
            }

            // irdft
            if (lane_id < block_size / 2 + 1) {
#pragma unroll
                for (int i = 0; i < block_size; i++) {
                    idft<2 * radius + 1>((float*)&thread_data[i], block_size);
                }
            }

            auto local_thread_data = &thread_data[radius * block_size];

            if (lane_id < block_size / 2 + 1) {
                __syncwarp(0x000001FF);
                idft<block_size>((float*)local_thread_data);

#pragma unroll
                for (int j = 0; j < block_size; j++) {
                    transpose_buffer[j * transpose_stride + lane_id] = local_thread_data[j];
                }
            }

            __syncwarp(active_mask);
#pragma unroll
            for (int j = 0; j < block_size / 2 + 1; j++) {
                local_thread_data[j].x = transpose_buffer[lane_id * transpose_stride + j].x;
                local_thread_data[j].y = transpose_buffer[lane_id * transpose_stride + j].y;
            }

            __syncwarp(active_mask);
            irdft<block_size>((float*)local_thread_data);

#pragma unroll
            for (int j = 0; j < block_size; j++) {
                ((float*)transpose_buffer)[j * transpose_stride + lane_id] =
                    ((float*)local_thread_data)[j == 0 ? j : block_size - j];
            }

            __syncwarp(active_mask);
            auto local_dst = &dstp[(block_id * (2 * radius + 1) + radius) * block_size * block_size];
#pragma unroll
            for (int j = 0; j < block_size; j++) {
                local_dst[j * block_size + lane_id] = ((float*)transpose_buffer)[lane_id * transpose_stride + j];
            }
        }
    }
}

template <int RADIUS, typename TYPE>
__device__ __forceinline__ void col2im_kernel(
    TYPE* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ window,
    float scale,
    float peak,
    int block_step,
    int width,
    int height
) {
    constexpr int radius = RADIUS;
    constexpr int block_size = 16;

    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int pad_x = (horizontal_size - width) / 2;
    int pad_y = (vertical_size - height) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < pad_y || y >= pad_y + height || x < pad_x || x >= pad_x + width) {
        return;
    }

    float sum = 0.0f;
    int i1 = (y - block_size + block_step) / block_step;
    int i2 = min(y / block_step, vertical_num - 1);
    int j1 = (x - block_size + block_step) / block_step;
    int j2 = min(x / block_step, horizontal_num - 1);

    for (int i = i1; i <= i2; i++) {
        int offset_y = y - i * block_step;
        for (int j = j1; j <= j2; j++) {
            int offset_x = x - j * block_step;
            auto src_offset =
                (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * block_size +
                offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = from_float<TYPE>(sum, scale, peak);
}

// Instantiate permutations
#define INSTANTIATE_FUSED(r, type, name, zm, zm_val)                                                                   \
    extern "C" __launch_bounds__(32) __global__ void fused_r##r##_##name##_zm##zm(                                     \
        float* __restrict__ dstp,                                                                                      \
        const type* __restrict__ srcp,                                                                                 \
        const float* __restrict__ window,                                                                              \
        const float* __restrict__ window_freq,                                                                         \
        const float* __restrict__ sigma_array,                                                                         \
        float sigma_scalar,                                                                                            \
        int sigma_is_scalar,                                                                                           \
        float sigma2,                                                                                                  \
        float pmin,                                                                                                    \
        float pmax,                                                                                                    \
        int filter_type,                                                                                               \
        float scale,                                                                                                   \
        int block_step,                                                                                                \
        int width,                                                                                                     \
        int height                                                                                                     \
    ) {                                                                                                                \
        fused_kernel<r, type, zm_val>(                                                                                 \
            dstp,                                                                                                      \
            srcp,                                                                                                      \
            window,                                                                                                    \
            window_freq,                                                                                               \
            sigma_array,                                                                                               \
            sigma_scalar,                                                                                              \
            sigma_is_scalar,                                                                                           \
            sigma2,                                                                                                    \
            pmin,                                                                                                      \
            pmax,                                                                                                      \
            filter_type,                                                                                               \
            scale,                                                                                                     \
            block_step,                                                                                                \
            width,                                                                                                     \
            height                                                                                                     \
        );                                                                                                             \
    }

#define INSTANTIATE_COL2IM(r, type, name)                                                                              \
    extern "C" __launch_bounds__(32) __global__ void col2im_r##r##_##name(                                             \
        type* __restrict__ dst,                                                                                        \
        const float* __restrict__ src,                                                                                 \
        const float* __restrict__ window,                                                                              \
        float scale,                                                                                                   \
        float peak,                                                                                                    \
        int block_step,                                                                                                \
        int width,                                                                                                     \
        int height                                                                                                     \
    ) {                                                                                                                \
        col2im_kernel<r, type>(dst, src, window, scale, peak, block_step, width, height);                              \
    }

#define INSTANTIATE_ALL_TYPES(r)                                                                                       \
    INSTANTIATE_FUSED(r, uint8_t, u8, 0, false)                                                                        \
    INSTANTIATE_FUSED(r, uint8_t, u8, 1, true)                                                                         \
    INSTANTIATE_FUSED(r, uint16_t, u16, 0, false)                                                                      \
    INSTANTIATE_FUSED(r, uint16_t, u16, 1, true)                                                                       \
    INSTANTIATE_FUSED(r, float, f32, 0, false)                                                                         \
    INSTANTIATE_FUSED(r, float, f32, 1, true)                                                                          \
    INSTANTIATE_COL2IM(r, uint8_t, u8)                                                                                 \
    INSTANTIATE_COL2IM(r, uint16_t, u16)                                                                               \
    INSTANTIATE_COL2IM(r, float, f32)

INSTANTIATE_ALL_TYPES(0)
INSTANTIATE_ALL_TYPES(1)
INSTANTIATE_ALL_TYPES(2)
INSTANTIATE_ALL_TYPES(3)
