#include <gtest/gtest.h>
#include "../include/compression.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

using namespace gpu_compression;

class CompressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        compressor = std::make_unique<Compressor>();
        compressor->enableProfiling(true);
    }

    std::unique_ptr<Compressor> compressor;
    
    // Helper to generate test data
    std::vector<uint32_t> generateTestData(size_t size, uint32_t max_value) {
        std::vector<uint32_t> data(size);
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<uint32_t> dist(0, max_value);
        
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen);
        }
        return data;
    }

    // Helper to generate RLE-friendly data
    std::vector<uint32_t> generateRLEData(size_t size) {
        std::vector<uint32_t> data(size);
        const int run_length = 16;
        
        for (size_t i = 0; i < size; i += run_length) {
            uint32_t value = (i / run_length) % 256;
            for (int j = 0; j < run_length && (i + j) < size; ++j) {
                data[i + j] = value;
            }
        }
        return data;
    }
};

TEST_F(CompressionTest, DictionaryCompression) {
    const size_t data_size = 1024 * 1024;
    auto input_data = generateTestData(data_size, 255);
    
    // Allocate device memory for input and output
    thrust::device_vector<uint32_t> d_input = input_data;
    thrust::device_vector<uint32_t> d_output(data_size);
    size_t output_size = data_size * sizeof(uint32_t);
    
    CompressionStats stats;
    auto status = compressor->compress(
        thrust::raw_pointer_cast(d_input.data()),
        data_size,
        thrust::raw_pointer_cast(d_output.data()),
        &output_size,
        &stats
    );
    
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_GT(stats.compression_ratio, 1.0f);
    ASSERT_LT(stats.compression_time_ms, 100.0f);
}

TEST_F(CompressionTest, RLECompression) {
    const size_t data_size = 1024 * 1024;
    auto input_data = generateRLEData(data_size);
    
    thrust::device_vector<uint32_t> d_input = input_data;
    thrust::device_vector<uint32_t> d_output(data_size * 2); // Space for values and counts
    size_t output_size = data_size * sizeof(uint32_t) * 2;
    
    compressor->setCompressionType(CompressionType::RLE);
    
    CompressionStats stats;
    auto status = compressor->compress(
        thrust::raw_pointer_cast(d_input.data()),
        data_size,
        thrust::raw_pointer_cast(d_output.data()),
        &output_size,
        &stats
    );
    
    ASSERT_EQ(status, cudaSuccess);
    ASSERT_GT(stats.compression_ratio, 8.0f); // RLE should achieve good compression for our test data
}

TEST_F(CompressionTest, CompressionDecompression) {
    const size_t data_size = 1024 * 1024;
    auto input_data = generateTestData(data_size, 255);
    
    thrust::device_vector<uint32_t> d_input = input_data;
    thrust::device_vector<uint32_t> d_compressed(data_size);
    thrust::device_vector<uint32_t> d_decompressed(data_size);
    
    size_t compressed_size = data_size * sizeof(uint32_t);
    size_t decompressed_size = data_size * sizeof(uint32_t);
    
    // Compress
    auto compress_status = compressor->compress(
        thrust::raw_pointer_cast(d_input.data()),
        data_size,
        thrust::raw_pointer_cast(d_compressed.data()),
        &compressed_size
    );
    
    ASSERT_EQ(compress_status, cudaSuccess);
    
    // Decompress
    auto decompress_status = compressor->decompress(
        thrust::raw_pointer_cast(d_compressed.data()),
        compressed_size,
        thrust::raw_pointer_cast(d_decompressed.data()),
        &decompressed_size
    );
    
    ASSERT_EQ(decompress_status, cudaSuccess);
    ASSERT_EQ(decompressed_size, data_size * sizeof(uint32_t));
    
    // Verify data
    thrust::host_vector<uint32_t> result = d_decompressed;
    for (size_t i = 0; i < data_size; ++i) {
        ASSERT_EQ(result[i], input_data[i]);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
