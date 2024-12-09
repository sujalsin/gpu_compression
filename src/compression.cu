#include "../include/compression.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace gpu_compression {

// Constants for GPU kernels
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_DICTIONARY_SIZE = 65536;

// Dictionary encoding kernel
__global__ void dictionaryEncodingKernel(
    const uint32_t* input,
    uint32_t* output,
    const uint32_t* dictionary,
    const uint32_t dict_size,
    const size_t input_size,
    uint32_t* indices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Shared memory for dictionary lookup
    __shared__ uint32_t shared_dict[256];
    
    // Load dictionary into shared memory
    for (int i = threadIdx.x; i < min(dict_size, 256u); i += blockDim.x) {
        shared_dict[i] = dictionary[i];
    }
    __syncthreads();

    // Process input data
    for (size_t i = tid; i < input_size; i += stride) {
        uint32_t value = input[i];
        uint32_t index = 0;
        
        // Binary search in shared dictionary
        int left = 0;
        int right = min(dict_size, 256u) - 1;
        
        while (left <= right) {
            int mid = (left + right) / 2;
            if (shared_dict[mid] == value) {
                index = mid;
                break;
            }
            else if (shared_dict[mid] < value) {
                left = mid + 1;
            }
            else {
                right = mid - 1;
            }
        }
        
        indices[i] = index;
    }
}

// Run-length encoding kernel
__global__ void rleEncodingKernel(
    const uint32_t* input,
    uint32_t* values,
    uint32_t* counts,
    const size_t input_size,
    uint32_t* output_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    __shared__ uint32_t shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    for (size_t i = tid; i < input_size; i += stride) {
        uint32_t current = input[i];
        uint32_t count = 1;
        
        // Count consecutive identical values
        while (i + 1 < input_size && input[i + 1] == current) {
            count++;
            i++;
        }
        
        // Atomically get position in output arrays
        uint32_t pos = atomicAdd(&shared_count, 1);
        
        // Store value and count
        if (pos < input_size) {
            values[pos] = current;
            counts[pos] = count;
        }
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(output_size, shared_count);
    }
}

// Advanced optimizations for dictionary compression
__global__ void buildAdaptiveDictionaryKernel(
    const uint32_t* input,
    uint32_t* dictionary,
    uint32_t* frequencies,
    const size_t input_size,
    const uint32_t dict_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Use shared memory for local frequency counting
    __shared__ uint32_t shared_freq[256];
    if (threadIdx.x < 256) {
        shared_freq[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Count frequencies
    for (size_t i = tid; i < input_size; i += stride) {
        uint32_t value = input[i] & 0xFF; // Consider only lower byte for frequency
        atomicAdd(&shared_freq[value], 1);
    }
    __syncthreads();
    
    // Update global frequencies
    if (threadIdx.x < 256) {
        atomicAdd(&frequencies[threadIdx.x], shared_freq[threadIdx.x]);
    }
}

// Decompression kernel for dictionary-based compression
__global__ void dictionaryDecompressionKernel(
    const uint32_t* compressed_data,
    const uint32_t* dictionary,
    uint32_t* output,
    const size_t compressed_size,
    const uint32_t dict_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Prefetch dictionary into L1 cache
    __shared__ uint32_t shared_dict[256];
    for (int i = threadIdx.x; i < min(dict_size, 256u); i += blockDim.x) {
        shared_dict[i] = dictionary[i];
    }
    __syncthreads();
    
    // Decompress data
    for (size_t i = tid; i < compressed_size; i += stride) {
        uint32_t index = compressed_data[i];
        if (index < dict_size) {
            output[i] = shared_dict[index];
        }
    }
}

// Stream compaction for RLE
__global__ void rleStreamCompactionKernel(
    const uint32_t* values,
    const uint32_t* counts,
    uint32_t* output,
    const uint32_t* prefix_sum,
    const size_t input_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < input_size) {
        uint32_t start_pos = (tid > 0) ? prefix_sum[tid - 1] : 0;
        uint32_t count = counts[tid];
        uint32_t value = values[tid];
        
        // Expand run-length encoded data
        for (uint32_t i = 0; i < count; ++i) {
            output[start_pos + i] = value;
        }
    }
}

Compressor::Compressor(CompressionType type)
    : type_(type)
    , compression_level_(6)
    , dictionary_size_(4096)
    , profiling_enabled_(false)
    , d_dictionary_(nullptr)
    , d_temp_storage_(nullptr)
    , temp_storage_size_(0)
{
    // Allocate GPU resources
    cudaMalloc(&d_dictionary_, dictionary_size_ * sizeof(uint32_t));
    cudaMalloc(&d_temp_storage_, 1024 * 1024); // Initial 1MB temp storage
}

Compressor::~Compressor() {
    if (d_dictionary_) cudaFree(d_dictionary_);
    if (d_temp_storage_) cudaFree(d_temp_storage_);
}

cudaError_t Compressor::compress(
    const void* input_data,
    size_t input_size,
    void* output_data,
    size_t* output_size,
    CompressionStats* stats
) {
    cudaError_t cuda_status = cudaSuccess;
    
    // Start timing if profiling is enabled
    cudaEvent_t start, stop;
    if (profiling_enabled_) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    const uint32_t* input = static_cast<const uint32_t*>(input_data);
    uint32_t* output = static_cast<uint32_t*>(output_data);
    
    // Calculate grid dimensions
    const int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (type_ == CompressionType::DICTIONARY) {
        // Allocate temporary storage for indices
        uint32_t* d_indices;
        cudaMalloc(&d_indices, input_size * sizeof(uint32_t));
        
        // Launch dictionary encoding kernel
        dictionaryEncodingKernel<<<num_blocks, BLOCK_SIZE>>>(
            input, output, static_cast<uint32_t*>(d_dictionary_),
            dictionary_size_, input_size / sizeof(uint32_t), d_indices
        );
        
        cudaFree(d_indices);
    }
    else if (type_ == CompressionType::RLE) {
        // Allocate temporary storage for RLE output
        uint32_t* d_values, *d_counts, *d_output_size;
        cudaMalloc(&d_values, input_size * sizeof(uint32_t));
        cudaMalloc(&d_counts, input_size * sizeof(uint32_t));
        cudaMalloc(&d_output_size, sizeof(uint32_t));
        
        // Initialize output size to 0
        cudaMemset(d_output_size, 0, sizeof(uint32_t));
        
        // Launch RLE encoding kernel
        rleEncodingKernel<<<num_blocks, BLOCK_SIZE>>>(
            input, d_values, d_counts, input_size / sizeof(uint32_t), d_output_size
        );
        
        // Copy results back
        uint32_t host_output_size;
        cudaMemcpy(&host_output_size, d_output_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        *output_size = host_output_size * sizeof(uint32_t) * 2;
        
        cudaFree(d_values);
        cudaFree(d_counts);
        cudaFree(d_output_size);
    }
    
    // Record compression stats if enabled
    if (profiling_enabled_ && stats) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        stats->compression_time_ms = milliseconds;
        stats->original_size = input_size;
        stats->compressed_size = *output_size;
        stats->compression_ratio = static_cast<float>(stats->original_size) / 
                                 static_cast<float>(stats->compressed_size);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return cuda_status;
}

cudaError_t Compressor::decompress(
    const void* input_data,
    size_t input_size,
    void* output_data,
    size_t* output_size,
    CompressionStats* stats
) {
    cudaError_t cuda_status = cudaSuccess;
    
    cudaEvent_t start, stop;
    if (profiling_enabled_) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }
    
    const uint32_t* input = static_cast<const uint32_t*>(input_data);
    uint32_t* output = static_cast<uint32_t*>(output_data);
    
    const int num_blocks = (*output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    if (type_ == CompressionType::DICTIONARY) {
        dictionaryDecompressionKernel<<<num_blocks, BLOCK_SIZE>>>(
            input,
            static_cast<uint32_t*>(d_dictionary_),
            output,
            input_size / sizeof(uint32_t),
            dictionary_size_
        );
    }
    else if (type_ == CompressionType::RLE) {
        // Allocate space for prefix sum
        thrust::device_vector<uint32_t> d_prefix_sum(input_size / sizeof(uint32_t) / 2);
        
        // Calculate prefix sum of counts
        thrust::inclusive_scan(
            thrust::device,
            input + input_size / sizeof(uint32_t) / 2,
            input + input_size / sizeof(uint32_t),
            thrust::raw_pointer_cast(d_prefix_sum.data())
        );
        
        rleStreamCompactionKernel<<<num_blocks, BLOCK_SIZE>>>(
            input,
            input + input_size / sizeof(uint32_t) / 2,
            output,
            thrust::raw_pointer_cast(d_prefix_sum.data()),
            input_size / sizeof(uint32_t) / 2
        );
    }
    
    if (profiling_enabled_ && stats) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        stats->decompression_time_ms = milliseconds;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    return cuda_status;
}

void Compressor::setCompressionLevel(int level) {
    compression_level_ = level;
}

void Compressor::setDictionarySize(size_t size) {
    dictionary_size_ = size;
    if (d_dictionary_) {
        cudaFree(d_dictionary_);
        cudaMalloc(&d_dictionary_, dictionary_size_ * sizeof(uint32_t));
    }
}

void Compressor::enableProfiling(bool enable) {
    profiling_enabled_ = enable;
}

void Compressor::updateDictionary(const void* input_data, size_t input_size) {
    const uint32_t* input = static_cast<const uint32_t*>(input_data);
    
    // Allocate frequency counter
    thrust::device_vector<uint32_t> d_frequencies(256, 0);
    
    const int num_blocks = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Build frequency table
    buildAdaptiveDictionaryKernel<<<num_blocks, BLOCK_SIZE>>>(
        input,
        static_cast<uint32_t*>(d_dictionary_),
        thrust::raw_pointer_cast(d_frequencies.data()),
        input_size,
        dictionary_size_
    );
    
    // Sort frequencies and update dictionary
    thrust::sort_by_key(
        thrust::device,
        d_frequencies.begin(),
        d_frequencies.end(),
        thrust::device_pointer_cast(static_cast<uint32_t*>(d_dictionary_)),
        thrust::greater<uint32_t>()
    );
}

} // namespace gpu_compression
