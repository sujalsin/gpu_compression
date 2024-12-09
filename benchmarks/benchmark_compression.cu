#include "../include/compression.h"
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace gpu_compression;

struct BenchmarkResult {
    float compression_throughput_gbps;
    float decompression_throughput_gbps;
    float compression_ratio;
    size_t input_size_bytes;
    size_t compressed_size_bytes;
};

class CompressionBenchmark {
public:
    CompressionBenchmark(size_t data_size_mb = 1024) 
        : data_size_(data_size_mb * 1024 * 1024 / sizeof(uint32_t)) {
        compressor_.enableProfiling(true);
    }

    BenchmarkResult runBenchmark(CompressionType type, int iterations = 10) {
        compressor_.setCompressionType(type);
        
        // Generate test data
        auto input_data = generateTestData();
        thrust::device_vector<uint32_t> d_input = input_data;
        thrust::device_vector<uint32_t> d_compressed(data_size_);
        thrust::device_vector<uint32_t> d_decompressed(data_size_);

        BenchmarkResult result{};
        result.input_size_bytes = data_size_ * sizeof(uint32_t);

        // Warmup
        size_t compressed_size = data_size_ * sizeof(uint32_t);
        compressor_.compress(
            thrust::raw_pointer_cast(d_input.data()),
            data_size_,
            thrust::raw_pointer_cast(d_compressed.data()),
            &compressed_size
        );

        // Benchmark compression
        CompressionStats stats;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            compressed_size = data_size_ * sizeof(uint32_t);
            compressor_.compress(
                thrust::raw_pointer_cast(d_input.data()),
                data_size_,
                thrust::raw_pointer_cast(d_compressed.data()),
                &compressed_size,
                &stats
            );
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Calculate throughput
        double total_gb = (static_cast<double>(data_size_) * sizeof(uint32_t) * iterations) / (1024 * 1024 * 1024);
        result.compression_throughput_gbps = total_gb / (duration / 1000.0);
        result.compression_ratio = static_cast<float>(data_size_ * sizeof(uint32_t)) / compressed_size;
        result.compressed_size_bytes = compressed_size;

        // Benchmark decompression
        size_t decompressed_size = data_size_ * sizeof(uint32_t);
        start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            compressor_.decompress(
                thrust::raw_pointer_cast(d_compressed.data()),
                compressed_size,
                thrust::raw_pointer_cast(d_decompressed.data()),
                &decompressed_size
            );
        }
        
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        result.decompression_throughput_gbps = total_gb / (duration / 1000.0);
        
        return result;
    }

private:
    Compressor compressor_;
    size_t data_size_;

    std::vector<uint32_t> generateTestData() {
        std::vector<uint32_t> data(data_size_);
        std::mt19937 gen(42);
        std::uniform_int_distribution<uint32_t> dist(0, 255);
        
        for (size_t i = 0; i < data_size_; ++i) {
            data[i] = dist(gen);
        }
        return data;
    }
};

int main() {
    // Test different data sizes
    const std::vector<size_t> sizes_mb = {64, 256, 1024, 4096};
    
    for (auto size_mb : sizes_mb) {
        std::cout << "\nBenchmarking with " << size_mb << "MB of data\n";
        CompressionBenchmark benchmark(size_mb);
        
        // Test dictionary compression
        auto dict_result = benchmark.runBenchmark(CompressionType::DICTIONARY);
        std::cout << "Dictionary Compression:\n"
                  << "  Compression Throughput: " << dict_result.compression_throughput_gbps << " GB/s\n"
                  << "  Decompression Throughput: " << dict_result.decompression_throughput_gbps << " GB/s\n"
                  << "  Compression Ratio: " << dict_result.compression_ratio << "\n";
        
        // Test RLE compression
        auto rle_result = benchmark.runBenchmark(CompressionType::RLE);
        std::cout << "RLE Compression:\n"
                  << "  Compression Throughput: " << rle_result.compression_throughput_gbps << " GB/s\n"
                  << "  Decompression Throughput: " << rle_result.decompression_throughput_gbps << " GB/s\n"
                  << "  Compression Ratio: " << rle_result.compression_ratio << "\n";
    }
    
    return 0;
}
