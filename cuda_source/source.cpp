#include <VSHelper4.h>
#include <VapourSynth4.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numbers>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef PLUGIN_VERSION_MAJOR
#define PLUGIN_VERSION_MAJOR 1
#endif
#ifndef PLUGIN_VERSION_MINOR
#define PLUGIN_VERSION_MINOR 0
#endif
#ifndef PLUGIN_VERSION_STRING
#define PLUGIN_VERSION_STRING "unknown"
#endif

#include <cuda.h>
#include <cufft.h>

#include "kernels_fatbin.h"

// real/complex-input DFT
template <typename T, typename T_in>
    requires(std::is_same_v<T_in, T> || std::is_same_v<T_in, std::complex<T>>)
static void dft(std::complex<T>* VS_RESTRICT dst, const T_in* VS_RESTRICT src, int n, int stride) {

    int out_num = std::is_floating_point_v<T_in> ? (n / 2 + 1) : n;
    for (int i = 0; i < out_num; i++) {
        std::complex<T> sum{};
        for (int j = 0; j < n; j++) {
            auto imag = -2 * i * j * std::numbers::pi_v<T> / n;
            auto weight = std::complex(std::cos(imag), std::sin(imag));
            sum += src[j * stride] * weight;
        }
        dst[i * stride] = sum;
    }
}

static bool success(CUresult result) {
    return result == CUDA_SUCCESS;
}
static bool success(cufftResult_t result) {
    return result == CUFFT_SUCCESS;
}

static const char* get_error(CUresult error) {
    const char* error_message;
    if (cuGetErrorString(error, &error_message) == CUDA_SUCCESS) [[likely]] {
        return error_message;
    }
    return "unknown cuda error";
}

static const char* get_error(cufftResult_t error) {
    switch (error) {
    case CUFFT_SUCCESS:
        return "success";
    case CUFFT_INVALID_PLAN:
        return "invalid plan handle";
    case CUFFT_ALLOC_FAILED:
        return "failed to allocate memory";
    case CUFFT_INVALID_TYPE:
        return "invalid type";
    case CUFFT_INVALID_VALUE:
        return "invalid value";
    case CUFFT_INTERNAL_ERROR:
        return "internal error";
    case CUFFT_EXEC_FAILED:
        return "execution failed";
    case CUFFT_SETUP_FAILED:
        return "the cuFFT library failed to initialize";
    case CUFFT_INVALID_SIZE:
        return "invalid transform size";
    case CUFFT_UNALIGNED_DATA:
        return "unaligned data";
#if CUFFT_VERSION < 12000
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "missing parameters in call";
#endif
    case CUFFT_INVALID_DEVICE:
        return "invalid device: execution of a plan was on different GPU than plan creation";
#if CUFFT_VERSION < 12000
    case CUFFT_PARSE_ERROR:
        return "internal plan database error";
#endif
    case CUFFT_NO_WORKSPACE:
        return "no workspace has been provided prior to plan execution";
    case CUFFT_NOT_IMPLEMENTED:
        return "functionality not implemented";
    case CUFFT_NOT_SUPPORTED:
        return "operation not supported";
#if CUFFT_VERSION >= 12000
    case CUFFT_MISSING_DEPENDENCY:
        return "missing dependency";
    case CUFFT_NVRTC_FAILURE:
        return "nvrtc failure";
    case CUFFT_NVJITLINK_FAILURE:
        return "nvjitlink failure";
    case CUFFT_NVSHMEM_FAILURE:
        return "nvshmem failure";
#endif
    default:
        return "unknown";
    }
}

#define showError(expr) show_error_impl(expr, #expr, __LINE__)
template <typename T> static void show_error_impl(T result, const char* source, int line_no) {
    if (!success(result)) [[unlikely]] {
        std::fprintf(stderr, "[%d] %s failed: %s\n", line_no, source, get_error(result));
    }
}

#define checkError(expr)                                                                                               \
    do {                                                                                                               \
        if (auto result = expr; !success(result)) [[unlikely]] {                                                       \
            std::ostringstream error;                                                                                  \
            error << '[' << __LINE__ << "] '" #expr "' failed: " << get_error(result);                                 \
            return set_error(error.str().c_str());                                                                     \
        }                                                                                                              \
    } while (0)

static void cuStreamDestroyCustom(CUstream stream) {
    showError(cuStreamDestroy(stream));
}

static void cuEventDestroyCustom(CUevent event) {
    showError(cuEventDestroy(event));
}

static void cuMemFreeCustom(CUdeviceptr p) {
    showError(cuMemFree(p));
}

static void cuMemFreeHostCustom(void* p) {
    showError(cuMemFreeHost(p));
}

static void cuModuleUnloadCustom(CUmodule module) {
    showError(cuModuleUnload(module));
}

static void cufftDestroyCustom(cufftHandle handle) {
    showError(cufftDestroy(handle));
}

struct context_releaser {
    bool* context_retained{};
    CUdevice device{};
    void release() { context_retained = nullptr; }
    ~context_releaser() {
        if (context_retained && *context_retained) {
            showError(cuDevicePrimaryCtxRelease(device));
        }
    }
};

struct context_popper {
    bool* context_pushed{};
    ~context_popper() {
        if (!context_pushed || *context_pushed) {
            showError(cuCtxPopCurrent(nullptr));
        }
    }
};

struct node_freer {
    const VSAPI*& vsapi;
    VSNode* node{};
    void release() { node = nullptr; }
    ~node_freer() {
        if (node) {
            vsapi->freeNode(node);
        }
    }
};

template <typename T, auto deleter, bool unsafe = false>
    requires std::default_initializable<T> && std::is_trivially_copy_assignable_v<T> && std::convertible_to<T, bool> &&
             std::invocable<decltype(deleter), T> && (std::is_pointer_v<T> || unsafe)
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;
    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}
    [[nodiscard]] constexpr Resource(Resource&& other) noexcept : data(std::exchange(other.data, T{})) {}

    Resource& operator=(Resource&& other) noexcept {
        if (this == &other)
            return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;
    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept { return data; }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
            x = T{};
        }
    }

    Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept { deleter_(data); }
};

template <typename T> static T square(const T& x) {
    return x * x;
}

static int calc_pad_size(int size, int block_size, int block_step) {
    return (
        size + ((size % block_size) ? block_size - size % block_size : 0) +
        std::max(block_size - block_step, block_step) * 2
    );
}

static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

template <typename T>
static void reflection_padding_impl(
    T* VS_RESTRICT dst,
    const T* VS_RESTRICT src,
    int width,
    int height,
    int stride,
    int block_size,
    int block_step
) {
    int pad_width = calc_pad_size(width, block_size, block_step);
    int pad_height = calc_pad_size(height, block_size, block_step);

    int offset_y = (pad_height - height) / 2;
    int offset_x = (pad_width - width) / 2;

    vsh::bitblt(
        &dst[offset_y * pad_width + offset_x], pad_width * sizeof(T), src, stride * sizeof(T), width * sizeof(T), height
    );

    for (int y = offset_y; y < offset_y + height; y++) {
        auto dst_line = &dst[y * pad_width];

        for (int x = 0; x < offset_x; x++) {
            dst_line[x] = dst_line[offset_x * 2 - x];
        }

        for (int x = offset_x + width; x < pad_width; x++) {
            dst_line[x] = dst_line[2 * (offset_x + width) - 2 - x];
        }
    }

    for (int y = 0; y < offset_y; y++) {
        std::memcpy(&dst[y * pad_width], &dst[(offset_y * 2 - y) * pad_width], pad_width * sizeof(T));
    }

    for (int y = offset_y + height; y < pad_height; y++) {
        std::memcpy(&dst[y * pad_width], &dst[(2 * (offset_y + height) - 2 - y) * pad_width], pad_width * sizeof(T));
    }
}

static void reflection_padding(
    uint8_t* VS_RESTRICT dst,
    const uint8_t* VS_RESTRICT src,
    int width,
    int height,
    int stride,
    int block_size,
    int block_step,
    int bytes_per_sample
) {
    if (bytes_per_sample == 1) {
        reflection_padding_impl(
            static_cast<uint8_t*>(dst), static_cast<const uint8_t*>(src), width, height, stride, block_size, block_step
        );
    } else if (bytes_per_sample == 2) {
        reflection_padding_impl(
            reinterpret_cast<uint16_t*>(dst),
            reinterpret_cast<const uint16_t*>(src),
            width,
            height,
            stride,
            block_size,
            block_step
        );
    } else if (bytes_per_sample == 4) {
        reflection_padding_impl(
            reinterpret_cast<uint32_t*>(dst),
            reinterpret_cast<const uint32_t*>(src),
            width,
            height,
            stride,
            block_size,
            block_step
        );
    }
}

struct DFTTestThreadData {
    uint8_t* h_padded;
};

struct DFTTestData {
    VSNode* node;
    int radius;
    int block_size;
    int block_step;
    std::array<bool, 3> process;
    CUdevice device;
    bool in_place;
    int zero_mean;
    int filter_type;
    float sigma_scalar;
    int sigma_is_scalar;
    float sigma2;
    float pmin;
    float pmax;

    int warp_size;
    int warps_per_block = 4;

    CUcontext context;
    Resource<CUstream, cuStreamDestroyCustom> stream;
    Resource<CUevent, cuEventDestroyCustom> event;

    // device buffers
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_window;
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_sigma;
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_window_freq;

    // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_spatial;

    // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size/2+1)
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_frequency;

    std::mutex lock;

    Resource<CUdeviceptr, cuMemFreeCustom, true> d_work_area_or_padded;

    Resource<cufftHandle, cufftDestroyCustom, true> rfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> irfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> subsampled_rfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> subsampled_irfft_handle;

    Resource<CUmodule, cuModuleUnloadCustom> module;
    CUfunction filter_kernel;
    int filter_num_blocks;
    CUfunction im2col_kernel;
    int im2col_num_blocks;
    CUfunction col2im_kernel;

    std::atomic<int> num_uninitialized_threads;
    std::unordered_map<std::thread::id, DFTTestThreadData> thread_data;
    std::shared_mutex thread_data_lock;
};

struct FrameFreer {
    const VSAPI* vsapi;
    void operator()(const VSFrame* f) const noexcept {
        if (f)
            vsapi->freeFrame(f);
    }
};

static const VSFrame* VS_CC DFTTestGetFrame(
    int n,
    int activationReason,
    void* instanceData,
    void** frameData,
    VSFrameContext* frameCtx,
    VSCore* core,
    const VSAPI* vsapi
) noexcept {

    auto d = static_cast<DFTTestData*>(instanceData);

    if (activationReason == arInitial) {
        int start = std::max(n - d->radius, 0);
        auto vi = vsapi->getVideoInfo(d->node);
        int end = std::min(n + d->radius, vi->numFrames - 1);
        for (int i = start; i <= end; i++) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        return nullptr;
    } else if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    auto set_error = [vsapi, frameCtx](const char* error_message) -> std::nullptr_t {
        vsapi->setFilterError(error_message, frameCtx);
        return nullptr;
    };

    checkError(cuCtxPushCurrent(d->context));
    context_popper context_popper;

    auto vi = vsapi->getVideoInfo(d->node);

    DFTTestThreadData thread_data;

    auto thread_id = std::this_thread::get_id();
    if (d->num_uninitialized_threads.load(std::memory_order_acquire) == 0) {
        const auto& const_data = d->thread_data;
        thread_data = const_data.at(thread_id);
    } else {
        bool initialized = true;

        d->thread_data_lock.lock_shared();
        try {
            const auto& const_data = d->thread_data;
            thread_data = const_data.at(thread_id);
        } catch (const std::out_of_range&) {
            initialized = false;
        }
        d->thread_data_lock.unlock_shared();

        if (!initialized) {
            auto padded_size =
                ((2 * d->radius + 1) * calc_pad_size(vi->height, d->block_size, d->block_step) *
                 calc_pad_size(vi->width, d->block_size, d->block_step) * vi->format.bytesPerSample);

            checkError(cuMemHostAlloc((void**)&thread_data.h_padded, padded_size, 0));

            {
                std::lock_guard _{d->thread_data_lock};
                d->thread_data.emplace(thread_id, thread_data);
            }

            d->num_uninitialized_threads.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::unique_ptr<const VSFrame, FrameFreer>> src_frames;
    src_frames.reserve(2 * d->radius + 1);
    for (int i = n - d->radius; i <= n + d->radius; i++) {
        src_frames.emplace_back(
            vsapi->getFrameFilter(std::clamp(i, 0, vi->numFrames - 1), d->node, frameCtx), FrameFreer{vsapi}
        );
    }

    auto src_center_frame = src_frames[d->radius].get();

    const VSFrame* fr[]{
        d->process[0] ? nullptr : src_center_frame,
        d->process[1] ? nullptr : src_center_frame,
        d->process[2] ? nullptr : src_center_frame
    };
    const int pl[]{0, 1, 2};
    std::unique_ptr<VSFrame, FrameFreer> dst_frame{
        vsapi->newVideoFrame2(&vi->format, vi->width, vi->height, fr, pl, src_center_frame, core), FrameFreer{vsapi}
    };

    for (int plane = 0; plane < vi->format.numPlanes; plane++) {
        if (!d->process[plane]) {
            continue;
        }

        int width = vsapi->getFrameWidth(src_center_frame, plane);
        int height = vsapi->getFrameHeight(src_center_frame, plane);
        int stride = vsapi->getStride(src_center_frame, plane) / vi->format.bytesPerSample;

        bool subsampled = vi->format.subSamplingW != 0 || vi->format.subSamplingH != 0;
        auto& rfft_handle = (plane == 0 || !subsampled) ? d->rfft_handle : d->subsampled_rfft_handle;
        auto& irfft_handle = (plane == 0 || !subsampled) ? d->irfft_handle : d->subsampled_irfft_handle;

        int padded_size_spatial =
            (calc_pad_size(height, d->block_size, d->block_step) * calc_pad_size(width, d->block_size, d->block_step));

        for (int i = 0; i < 2 * d->radius + 1; i++) {
            auto srcp = vsapi->getReadPtr(src_frames[i].get(), plane);
            reflection_padding(
                &thread_data.h_padded[(i * padded_size_spatial) * vi->format.bytesPerSample],
                srcp,
                width,
                height,
                stride,
                d->block_size,
                d->block_step,
                vi->format.bytesPerSample
            );
        }

        {
            std::lock_guard lock{d->lock};

            CUdeviceptr d_buffer = d->in_place ? d->d_frequency.data : d->d_spatial.data;
            int padded_block_size = d->in_place ? (d->block_size / 2 + 1) * 2 : d->block_size;

            int padded_bytes = (2 * d->radius + 1) * padded_size_spatial * vi->format.bytesPerSample;
            checkError(cuMemcpyHtoDAsync(d->d_work_area_or_padded.data, thread_data.h_padded, padded_bytes, d->stream));

            float scale = (vi->format.sampleType == stInteger)
                              ? static_cast<float>(1.0 / (1 << (vi->format.bitsPerSample - 8)))
                              : 255.0f;
            float peak = static_cast<float>((1 << vi->format.bitsPerSample) - 1);

            {
                void* params[]{
                    &d_buffer,
                    &d->d_work_area_or_padded.data,
                    &d->d_window.data,
                    &scale,
                    &d->radius,
                    &d->block_size,
                    &d->block_step,
                    &padded_block_size,
                    &width,
                    &height
                };
                checkError(cuLaunchKernel(
                    d->im2col_kernel,
                    d->im2col_num_blocks,
                    1,
                    1,
                    d->warps_per_block * d->warp_size,
                    1,
                    1,
                    0,
                    d->stream,
                    params,
                    nullptr
                ));
            }
            checkError(cufftExecR2C(rfft_handle, (cufftReal*)d_buffer, (cufftComplex*)d->d_frequency.data));
            {
                int num_blocks =
                    (calc_pad_num(height, d->block_size, d->block_step) *
                     calc_pad_num(width, d->block_size, d->block_step));
                void* params[]{
                    &d->d_frequency.data,
                    &num_blocks,
                    &d->radius,
                    &d->block_size,
                    &d->d_window_freq.data,
                    &d->d_sigma.data,
                    &d->sigma_scalar,
                    &d->sigma_is_scalar,
                    &d->sigma2,
                    &d->pmin,
                    &d->pmax,
                    &d->filter_type
                };
                checkError(cuLaunchKernel(
                    d->filter_kernel,
                    d->filter_num_blocks,
                    1,
                    1,
                    d->warps_per_block * d->warp_size,
                    1,
                    1,
                    0,
                    d->stream,
                    params,
                    nullptr
                ));
            }
            checkError(cufftExecC2R(irfft_handle, (cufftComplex*)d->d_frequency.data, (cufftReal*)d_buffer));
            {
                void* params[]{
                    &d->d_work_area_or_padded.data,
                    &d_buffer,
                    &d->d_window.data,
                    &scale,
                    &peak,
                    &d->radius,
                    &d->block_size,
                    &d->block_step,
                    &padded_block_size,
                    &width,
                    &height
                };
                unsigned int vertical_size = calc_pad_size(height, d->block_size, d->block_step);
                unsigned int horizontal_size = calc_pad_size(width, d->block_size, d->block_step);
                unsigned int grid_x = (horizontal_size + d->warp_size - 1) / d->warp_size;
                unsigned int grid_y = (vertical_size + d->warps_per_block - 1) / d->warps_per_block;
                checkError(cuLaunchKernel(
                    d->col2im_kernel,
                    grid_x,
                    grid_y,
                    1,
                    d->warp_size,
                    d->warps_per_block,
                    1,
                    0,
                    d->stream,
                    params,
                    nullptr
                ));
            }
            {
                size_t pad_width = calc_pad_size(width, d->block_size, d->block_step);
                size_t pad_height = calc_pad_size(height, d->block_size, d->block_step);
                const CUDA_MEMCPY3D config{
                    .srcXInBytes = (pad_width - width) / 2 * vi->format.bytesPerSample,
                    .srcY = (pad_height - height) / 2,
                    .srcZ = (size_t)d->radius,
                    .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                    .srcDevice = d->d_work_area_or_padded.data,
                    .srcPitch = pad_width * vi->format.bytesPerSample,
                    .srcHeight = pad_height,
                    .dstXInBytes = (pad_width - width) / 2 * vi->format.bytesPerSample,
                    .dstY = (pad_height - height) / 2,
                    .dstZ = 0,
                    .dstMemoryType = CU_MEMORYTYPE_HOST,
                    .dstHost = thread_data.h_padded,
                    .dstPitch = pad_width * vi->format.bytesPerSample,
                    .dstHeight = pad_height,
                    .WidthInBytes = (size_t)width * vi->format.bytesPerSample,
                    .Height = (size_t)height,
                    .Depth = 1
                };
                checkError(cuMemcpy3DAsync(&config, d->stream));
            }

            checkError(cuEventRecord(d->event, d->stream));
            checkError(cuEventSynchronize(d->event));
        }

        int pad_width = calc_pad_size(width, d->block_size, d->block_step);
        int pad_height = calc_pad_size(height, d->block_size, d->block_step);
        int offset_y = (pad_height - height) / 2;
        int offset_x = (pad_width - width) / 2;

        auto dstp = vsapi->getWritePtr(dst_frame.get(), plane);
        auto input = &thread_data.h_padded[(offset_y * pad_width + offset_x) * vi->format.bytesPerSample];
        vsh::bitblt(
            dstp,
            stride * vi->format.bytesPerSample,
            input,
            pad_width * vi->format.bytesPerSample,
            width * vi->format.bytesPerSample,
            height
        );
    }

    return dst_frame.release();
}

static void VS_CC DFTTestFree(void* instanceData, VSCore* core, const VSAPI* vsapi) noexcept {

    auto d = static_cast<DFTTestData*>(instanceData);

    vsapi->freeNode(d->node);

    showError(cuCtxPushCurrent(d->context));

    for (const auto& [_, thread_data] : d->thread_data) {
        showError(cuMemFreeHost(thread_data.h_padded));
    }

    auto device = d->device;

    delete d;

    showError(cuCtxPopCurrent(nullptr));

    showError(cuDevicePrimaryCtxRelease(device));
}

static void VS_CC
DFTTestCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) noexcept {

    bool context_retained = false;
    bool context_pushed = false;

    context_releaser context_releaser{&context_retained};
    context_popper context_popper{&context_pushed};

    auto d = std::make_unique<DFTTestData>();

    d->node = vsapi->mapGetNode(in, "clip", 0, nullptr);
    node_freer node_freer{vsapi, d->node};

    auto set_error = [vsapi, out](const char* error_message) -> void { vsapi->mapSetError(out, error_message); };

    auto vi = vsapi->getVideoInfo(d->node);
    if (!vsh::isConstantVideoFormat(vi)) {
        return set_error("only constant format input is supported");
    }

    int error = 0;

    d->radius = vsh::int64ToIntS(vsapi->mapGetInt(in, "radius", 0, &error));
    if (error) {
        d->radius = 0;
    }

    d->block_size = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_size", 0, &error));
    if (error) {
        d->block_size = 8;
    }

    d->block_step = vsh::int64ToIntS(vsapi->mapGetInt(in, "block_step", 0, &error));
    if (error) {
        d->block_step = d->block_size;
    }

    d->zero_mean = vsh::int64ToIntS(vsapi->mapGetInt(in, "zero_mean", 0, &error));
    if (error) {
        d->zero_mean = 0;
    }

    d->filter_type = vsh::int64ToIntS(vsapi->mapGetInt(in, "filter_type", 0, &error));
    if (error) {
        d->filter_type = 0;
    }

    d->sigma2 = static_cast<float>(vsapi->mapGetFloat(in, "sigma2", 0, &error));
    if (error) {
        d->sigma2 = 8.0f;
    }

    d->pmin = static_cast<float>(vsapi->mapGetFloat(in, "pmin", 0, &error));
    if (error) {
        d->pmin = 0.0f;
    }

    d->pmax = static_cast<float>(vsapi->mapGetFloat(in, "pmax", 0, &error));
    if (error) {
        d->pmax = 500.0f;
    }

    int num_planes_args = vsapi->mapNumElements(in, "planes");
    d->process.fill(num_planes_args <= 0);
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = static_cast<int>(vsapi->mapGetInt(in, "planes", i, nullptr));

        if (plane < 0 || plane >= vi->format.numPlanes) {
            return set_error("plane index out of range");
        }

        if (d->process[plane]) {
            return set_error("plane specified twice");
        }

        d->process[plane] = true;
    }

    d->in_place = !!(vsapi->mapGetInt(in, "in_place", 0, &error));
    if (error) {
        d->in_place = true;
    }

    int device_id = vsh::int64ToIntS(vsapi->mapGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    // Read window array
    int num_window = vsapi->mapNumElements(in, "window");
    if (num_window <= 0) {
        return set_error("window cannot be empty");
    }
    const double* window_raw = vsapi->mapGetFloatArray(in, "window", nullptr);
    std::vector<float> window_floats(num_window);
    for (int i = 0; i < num_window; ++i) {
        window_floats[i] = static_cast<float>(window_raw[i]);
    }

    // Read sigma array
    int num_sigma = vsapi->mapNumElements(in, "sigma");
    if (num_sigma <= 0) {
        return set_error("sigma cannot be empty");
    }
    const double* sigma_raw = vsapi->mapGetFloatArray(in, "sigma", nullptr);
    std::vector<float> sigma_floats(num_sigma);
    for (int i = 0; i < num_sigma; ++i) {
        sigma_floats[i] = static_cast<float>(sigma_raw[i]);
    }
    if (num_sigma == 1) {
        d->sigma_scalar = sigma_floats[0];
        d->sigma_is_scalar = 1;
    } else {
        d->sigma_scalar = 0.0f;
        d->sigma_is_scalar = 0;
    }

    // Read window_freq if zero_mean enabled
    std::vector<float> window_freq_floats;
    if (d->zero_mean) {
        int num_wf = vsapi->mapNumElements(in, "window_freq");
        if (num_wf <= 0) {
            return set_error("window_freq required when zero_mean is enabled");
        }
        const double* wf_raw = vsapi->mapGetFloatArray(in, "window_freq", nullptr);
        window_freq_floats.resize(num_wf);
        for (int i = 0; i < num_wf; ++i) {
            window_freq_floats[i] = static_cast<float>(wf_raw[i]);
        }
    }

    checkError(cuInit(0));
    checkError(cuDeviceGet(&d->device, device_id));

    checkError(cuDevicePrimaryCtxRetain(&d->context, d->device));
    context_retained = true;
    context_releaser.device = d->device;

    checkError(cuCtxPushCurrent(d->context));
    context_pushed = true;

    checkError(cuDeviceGetAttribute(&d->warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, d->device));

    // Allocate & copy window buffer
    checkError(cuMemAlloc(&d->d_window.data, window_floats.size() * sizeof(float)));
    checkError(cuMemcpyHtoD(d->d_window.data, window_floats.data(), window_floats.size() * sizeof(float)));

    // Allocate & copy sigma buffer if array
    if (!d->sigma_is_scalar) {
        checkError(cuMemAlloc(&d->d_sigma.data, sigma_floats.size() * sizeof(float)));
        checkError(cuMemcpyHtoD(d->d_sigma.data, sigma_floats.data(), sigma_floats.size() * sizeof(float)));
    }

    // Allocate & copy window_freq buffer if zero_mean
    if (d->zero_mean) {
        checkError(cuMemAlloc(&d->d_window_freq.data, window_freq_floats.size() * sizeof(float)));
        checkError(
            cuMemcpyHtoD(d->d_window_freq.data, window_freq_floats.data(), window_freq_floats.size() * sizeof(float))
        );
    }

    // Load precompiled Fatbin module
    checkError(cuModuleLoadData(&d->module.data, kernels_fatbin));

    const char* type_str = (vi->format.sampleType == stFloat) ? "f32" : (vi->format.bytesPerSample == 1 ? "u8" : "u16");
    std::string im2col_name = std::string("im2col_") + type_str;
    std::string filter_name = std::string("frequency_filtering_zm") + (d->zero_mean ? "1" : "0");
    std::string col2im_name = std::string("col2im_") + type_str;

    checkError(cuModuleGetFunction(&d->im2col_kernel, d->module, im2col_name.c_str()));
    checkError(cuModuleGetFunction(&d->filter_kernel, d->module, filter_name.c_str()));
    checkError(cuModuleGetFunction(&d->col2im_kernel, d->module, col2im_name.c_str()));

    int num_sms;
    checkError(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d->device));

    int max_blocks_per_sm;
    checkError(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, d->filter_kernel, d->warps_per_block * d->warp_size, 0
    ));
    d->filter_num_blocks = num_sms * max_blocks_per_sm;

    checkError(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, d->im2col_kernel, d->warps_per_block * d->warp_size, 0
    ));
    d->im2col_num_blocks = num_sms * max_blocks_per_sm;

    checkError(cuStreamCreate(&d->stream.data, CU_STREAM_NON_BLOCKING));

    checkError(cuEventCreate(&d->event.data, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));

    size_t padded_bytes =
        ((2 * d->radius + 1) * calc_pad_size(vi->height, d->block_size, d->block_step) *
         calc_pad_size(vi->width, d->block_size, d->block_step) * vi->format.bytesPerSample);

    if (!d->in_place) {
        size_t spatial_bytes =
            (calc_pad_num(vi->height, d->block_size, d->block_step) *
             calc_pad_num(vi->width, d->block_size, d->block_step) * (2 * d->radius + 1) * square(d->block_size) *
             sizeof(cufftReal));
        checkError(cuMemAlloc(&d->d_spatial.data, spatial_bytes));
    }

    size_t frequency_bytes =
        ((2 * d->radius + 1) * calc_pad_num(vi->height, d->block_size, d->block_step) *
         calc_pad_num(vi->width, d->block_size, d->block_step) * d->block_size * (d->block_size / 2 + 1) *
         sizeof(cufftComplex));
    checkError(cuMemAlloc(&d->d_frequency.data, frequency_bytes));

    // init cufft
    {
        size_t max_work_size{padded_bytes};

        int batch =
            (calc_pad_num(vi->height, d->block_size, d->block_step) *
             calc_pad_num(vi->width, d->block_size, d->block_step));
        std::array<int, 3> fft_size{2 * d->radius + 1, d->block_size, d->block_size};
        int is_spatial = d->radius == 0;
        int fft_rank = 3 - is_spatial;
        auto fft_n = &fft_size[is_spatial];

        checkError(
            cufftPlanMany(&d->rfft_handle.data, fft_rank, fft_n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, batch)
        );
        {
            size_t work_size;
            checkError(cufftGetSize(d->rfft_handle, &work_size));
            max_work_size = std::max(max_work_size, work_size);
        }
        checkError(cufftSetWorkArea(d->rfft_handle, nullptr));
        checkError(cufftSetStream(d->rfft_handle, d->stream));

        checkError(
            cufftPlanMany(&d->irfft_handle.data, fft_rank, fft_n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, batch)
        );
        {
            size_t work_size;
            checkError(cufftGetSize(d->irfft_handle, &work_size));
            max_work_size = std::max(max_work_size, work_size);
        }
        checkError(cufftSetWorkArea(d->irfft_handle, nullptr));
        checkError(cufftSetStream(d->irfft_handle, d->stream));

        if (vi->format.subSamplingW != 0 || vi->format.subSamplingH != 0) {
            int subsampled_batch =
                (calc_pad_num(vi->height >> vi->format.subSamplingH, d->block_size, d->block_step) *
                 calc_pad_num(vi->width >> vi->format.subSamplingW, d->block_size, d->block_step));

            checkError(cufftPlanMany(
                &d->subsampled_rfft_handle.data,
                fft_rank,
                fft_n,
                nullptr,
                1,
                0,
                nullptr,
                1,
                0,
                CUFFT_R2C,
                subsampled_batch
            ));
            {
                size_t work_size;
                checkError(cufftGetSize(d->subsampled_rfft_handle, &work_size));
                max_work_size = std::max(max_work_size, work_size);
            }
            checkError(cufftSetWorkArea(d->subsampled_rfft_handle, nullptr));
            checkError(cufftSetStream(d->subsampled_rfft_handle, d->stream));

            checkError(cufftPlanMany(
                &d->subsampled_irfft_handle.data,
                fft_rank,
                fft_n,
                nullptr,
                1,
                0,
                nullptr,
                1,
                0,
                CUFFT_C2R,
                subsampled_batch
            ));
            {
                size_t work_size;
                checkError(cufftGetSize(d->subsampled_irfft_handle, &work_size));
                max_work_size = std::max(max_work_size, work_size);
            }
            checkError(cufftSetWorkArea(d->subsampled_irfft_handle, nullptr));
            checkError(cufftSetStream(d->subsampled_irfft_handle, d->stream));
        }

        checkError(cuMemAlloc(&d->d_work_area_or_padded.data, max_work_size));
        checkError(cufftSetWorkArea(d->rfft_handle, (void*)d->d_work_area_or_padded.data));
        checkError(cufftSetWorkArea(d->irfft_handle, (void*)d->d_work_area_or_padded.data));
        if (vi->format.subSamplingW != 0 || vi->format.subSamplingH != 0) {
            checkError(cufftSetWorkArea(d->subsampled_rfft_handle, (void*)d->d_work_area_or_padded.data));
            checkError(cufftSetWorkArea(d->subsampled_irfft_handle, (void*)d->d_work_area_or_padded.data));
        }
    }

    VSCoreInfo info;
    vsapi->getCoreInfo(core, &info);
    d->num_uninitialized_threads.store(info.numThreads, std::memory_order_relaxed);
    d->thread_data.reserve(info.numThreads);

    VSFilterDependency deps[] = {{d->node, rpGeneral}};
    VSVideoInfo out_vi = *vi;

    vsapi->createVideoFilter(
        out, "DFTTest", &out_vi, DFTTestGetFrame, DFTTestFree, fmParallel, deps, 1, d.release(), core
    );

    node_freer.release();
    context_releaser.release();
}

static void VS_CC RDFT(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) noexcept {

    auto set_error = [vsapi, out](const char* error_message) -> void { vsapi->mapSetError(out, error_message); };

    int ndim = vsapi->mapNumElements(in, "shape");
    if (ndim != 1 && ndim != 2 && ndim != 3) {
        return set_error("\"shape\" must be an array of ints with 1, 2 or 3 values");
    }

    std::array<int, 3> shape{};
    {
        auto shape_array = vsapi->mapGetIntArray(in, "shape", nullptr);
        for (int i = 0; i < ndim; i++) {
            shape[i] = vsh::int64ToIntS(shape_array[i]);
        }
    }

    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    if (vsapi->mapNumElements(in, "data") != size) {
        return set_error("cannot reshape array");
    }

    int complex_size = shape[ndim - 1] / 2 + 1;
    for (int i = 0; i < ndim - 1; i++) {
        complex_size *= shape[i];
    }

    auto input = vsapi->mapGetFloatArray(in, "data", nullptr);

    auto output = std::make_unique<std::complex<double>[]>(complex_size);

    if (ndim == 1) {
        dft(output.get(), input, size, 1);
        vsapi->mapSetFloatArray(out, "ret", (const double*)output.get(), complex_size * 2);
    } else if (ndim == 2) {
        for (int i = 0; i < shape[0]; i++) {
            dft(&output[i * (shape[1] / 2 + 1)], &input[i * shape[1]], shape[1], 1);
        }

        auto output2 = std::make_unique<std::complex<double>[]>(complex_size);

        for (int i = 0; i < shape[1] / 2 + 1; i++) {
            dft(&output2[i], &output[i], shape[0], shape[1] / 2 + 1);
        }

        vsapi->mapSetFloatArray(out, "ret", (const double*)output2.get(), complex_size * 2);
    } else {
        for (int i = 0; i < shape[0] * shape[1]; i++) {
            dft(&output[i * (shape[2] / 2 + 1)], &input[i * shape[2]], shape[2], 1);
        }

        auto output2 = std::make_unique<std::complex<double>[]>(complex_size);

        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[2] / 2 + 1; j++) {
                dft(&output2[i * shape[1] * (shape[2] / 2 + 1) + j],
                    &output[i * shape[1] * (shape[2] / 2 + 1) + j],
                    shape[1],
                    (shape[2] / 2 + 1));
            }
        }

        for (int i = 0; i < shape[1] * (shape[2] / 2 + 1); i++) {
            dft(&output[i], &output2[i], shape[0], shape[1] * (shape[2] / 2 + 1));
        }

        vsapi->mapSetFloatArray(out, "ret", (const double*)output.get(), complex_size * 2);
    }
}

static void Version(const VSMap*, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
    vsapi->mapSetData(out, "version", PLUGIN_VERSION_STRING, -1, dtUtf8, maReplace);

    vsapi->mapSetInt(out, "cufft_version_build", CUFFT_VERSION, maReplace);

    int cufft_version;
    if (cufftGetVersion(&cufft_version) == CUFFT_SUCCESS) {
        vsapi->mapSetInt(out, "cufft_version", cufft_version, maReplace);
    }
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(
        "io.github.amusementclub.dfttest2_cuda",
        "dfttest2_cuda",
        "DFTTest2 (CUDA)",
        VS_MAKE_VERSION(PLUGIN_VERSION_MAJOR, PLUGIN_VERSION_MINOR),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );

    vspapi->registerFunction(
        "DFTTest",
        "clip:vnode;"
        "window:float[];"
        "sigma:float[];"
        "sigma2:float;"
        "pmin:float;"
        "pmax:float;"
        "filter_type:int;"
        "radius:int:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "zero_mean:int:opt;"
        "window_freq:float[]:opt;"
        "planes:int[]:opt;"
        "in_place:int:opt;"
        "device_id:int:opt;",
        "clip:vnode;",
        DFTTestCreate,
        nullptr,
        plugin
    );

    vspapi->registerFunction(
        "RDFT",
        "data:float[];"
        "shape:int[];",
        "window_freq:float[];",
        RDFT,
        nullptr,
        plugin
    );

    vspapi->registerFunction("Version", "", "any", Version, nullptr, plugin);
}
