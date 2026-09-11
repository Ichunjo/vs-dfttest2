#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

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

template <typename TYPE>
__device__ __forceinline__ void im2col_impl(
    float* __restrict__ dstp,
    const TYPE* __restrict__ srcp,
    const float* __restrict__ window,
    float scale,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    constexpr int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int warps_per_block = blockDim.x / warp_size;

    for (int i = blockIdx.x * warps_per_block + warp_id; i < num_blocks; i += gridDim.x * warps_per_block) {
        int ix = i % horizontal_num;
        int iy = i / horizontal_num;
        auto dst = &dstp[i * (2 * radius + 1) * block_size * padded_block_size];
        for (int j = 0; j < 2 * radius + 1; j++) {
            auto src = &srcp[(j * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
            for (int k = lane_id; k < block_size * block_size; k += warp_size) {
                int kx = k % block_size;
                int ky = k / block_size;
                float val = to_float(src[ky * horizontal_size + kx], scale) * window[j * block_size * block_size + k];
                dst[(j * block_size + ky) * padded_block_size + kx] = val;
            }
        }
    }
}

template <bool ZERO_MEAN>
__device__ __forceinline__ void frequency_filtering_impl(
    float2* __restrict__ data,
    int num_blocks,
    int radius,
    int block_size_1d,
    const float* __restrict__ window_freq,
    const float* __restrict__ sigma_array,
    float sigma_scalar,
    int sigma_is_scalar,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type
) {
    int block_size_x = block_size_1d / 2 + 1;
    int block_size_2d = block_size_1d * block_size_x;
    int block_size_3d = (2 * radius + 1) * block_size_2d;

    constexpr int warp_size = 32;
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int warps_per_block = blockDim.x / warp_size;

    for (int i = blockIdx.x * warps_per_block + warp_id; i < num_blocks; i += gridDim.x * warps_per_block) {
        [[maybe_unused]] float gf = 0.0f;
        if constexpr (ZERO_MEAN) {
            if (lane_id == 0) {
                gf = data[i * block_size_3d].x / window_freq[0];
            }
            gf = __shfl_sync(0xFFFFFFFF, gf, 0);
        }

        for (int j = lane_id; j < block_size_3d; j += warp_size) {
            float2 local_data = data[i * block_size_3d + j];

            [[maybe_unused]] float val1 = 0.0f;
            [[maybe_unused]] float val2 = 0.0f;
            if constexpr (ZERO_MEAN) {
                // remove mean
                val1 = gf * window_freq[j * 2];
                val2 = gf * window_freq[j * 2 + 1];
                local_data.x -= val1;
                local_data.y -= val2;
            }

            int x = j % block_size_x;
            int y = (j % block_size_2d) / block_size_x;
            int t = (j % block_size_3d) / block_size_2d;

            apply_filter(
                local_data,
                x,
                y,
                t,
                sigma_array,
                sigma_scalar,
                sigma_is_scalar,
                sigma2,
                pmin,
                pmax,
                filter_type,
                block_size_1d
            );

            if constexpr (ZERO_MEAN) {
                local_data.x += val1;
                local_data.y += val2;
            }

            data[i * block_size_3d + j] = local_data;
        }
    }
}

template <typename TYPE>
__device__ __forceinline__ void col2im_impl(
    TYPE* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ window,
    float scale,
    float peak,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
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
                (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * padded_block_size +
                offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = from_float<TYPE>(sum, scale, peak);
}

// im2col exports
extern "C" __launch_bounds__(128) __global__ void im2col_u8(
    float* __restrict__ dstp,
    const uint8_t* __restrict__ srcp,
    const float* __restrict__ window,
    float scale,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    im2col_impl<uint8_t>(dstp, srcp, window, scale, radius, block_size, block_step, padded_block_size, width, height);
}

extern "C" __launch_bounds__(128) __global__ void im2col_u16(
    float* __restrict__ dstp,
    const uint16_t* __restrict__ srcp,
    const float* __restrict__ window,
    float scale,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    im2col_impl<uint16_t>(dstp, srcp, window, scale, radius, block_size, block_step, padded_block_size, width, height);
}

extern "C" __launch_bounds__(128) __global__ void im2col_f32(
    float* __restrict__ dstp,
    const float* __restrict__ srcp,
    const float* __restrict__ window,
    float scale,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    im2col_impl<float>(dstp, srcp, window, scale, radius, block_size, block_step, padded_block_size, width, height);
}

// frequency_filtering exports
extern "C" __launch_bounds__(128) __global__ void frequency_filtering_zm0(
    float2* __restrict__ data,
    int num_blocks,
    int radius,
    int block_size_1d,
    const float* __restrict__ window_freq,
    const float* __restrict__ sigma_array,
    float sigma_scalar,
    int sigma_is_scalar,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type
) {
    frequency_filtering_impl<false>(
        data,
        num_blocks,
        radius,
        block_size_1d,
        window_freq,
        sigma_array,
        sigma_scalar,
        sigma_is_scalar,
        sigma2,
        pmin,
        pmax,
        filter_type
    );
}

extern "C" __launch_bounds__(128) __global__ void frequency_filtering_zm1(
    float2* __restrict__ data,
    int num_blocks,
    int radius,
    int block_size_1d,
    const float* __restrict__ window_freq,
    const float* __restrict__ sigma_array,
    float sigma_scalar,
    int sigma_is_scalar,
    float sigma2,
    float pmin,
    float pmax,
    int filter_type
) {
    frequency_filtering_impl<true>(
        data,
        num_blocks,
        radius,
        block_size_1d,
        window_freq,
        sigma_array,
        sigma_scalar,
        sigma_is_scalar,
        sigma2,
        pmin,
        pmax,
        filter_type
    );
}

// col2im exports
extern "C" __launch_bounds__(128) __global__ void col2im_u8(
    uint8_t* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ window,
    float scale,
    float peak,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    col2im_impl<uint8_t>(
        dst, src, window, scale, peak, radius, block_size, block_step, padded_block_size, width, height
    );
}

extern "C" __launch_bounds__(128) __global__ void col2im_u16(
    uint16_t* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ window,
    float scale,
    float peak,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    col2im_impl<uint16_t>(
        dst, src, window, scale, peak, radius, block_size, block_step, padded_block_size, width, height
    );
}

extern "C" __launch_bounds__(128) __global__ void col2im_f32(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ window,
    float scale,
    float peak,
    int radius,
    int block_size,
    int block_step,
    int padded_block_size,
    int width,
    int height
) {
    col2im_impl<float>(dst, src, window, scale, peak, radius, block_size, block_step, padded_block_size, width, height);
}
