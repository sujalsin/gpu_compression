#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace gpu_compression {

enum class CompressionType {
    DICTIONARY,
    RLE,
    HYBRID
};

struct CompressionStats {
    size_t original_size;
    size_t compressed_size;
    float compression_ratio;
    float compression_time_ms;
    float decompression_time_ms;
};

class Compressor {
public:
    Compressor(CompressionType type = CompressionType::DICTIONARY);
    ~Compressor();

    // Compress data on GPU
    cudaError_t compress(const void* input_data, size_t input_size,
                        void* output_data, size_t* output_size,
                        CompressionStats* stats = nullptr);

    // Decompress data on GPU
    cudaError_t decompress(const void* input_data, size_t input_size,
                          void* output_data, size_t* output_size,
                          CompressionStats* stats = nullptr);

    // Configure compression parameters
    void setCompressionLevel(int level);
    void setDictionarySize(size_t size);
    void enableProfiling(bool enable);
    void setCompressionType(CompressionType type) { type_ = type; }
    
    // Advanced optimization methods
    void updateDictionary(const void* input_data, size_t input_size);

private:
    CompressionType type_;
    int compression_level_;
    size_t dictionary_size_;
    bool profiling_enabled_;

    // GPU resources
    void* d_dictionary_;
    void* d_temp_storage_;
    size_t temp_storage_size_;
    
    // Stream handling for async operations
    cudaStream_t stream_;
};

} // namespace gpu_compression
