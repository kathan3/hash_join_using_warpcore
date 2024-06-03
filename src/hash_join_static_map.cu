#include <iostream>
#include <warpcore/single_value_hash_table.cuh>

int main()
{
    // Key type of the hash table (uint32_t or uint64_t)
    using key_t = std::uint32_t;

    // Value type of the hash table
    using value_t = std::uint32_t;

    using namespace warpcore;

    // Type of the hash table (with default parameters)
    using hash_table_t = SingleValueHashTable<key_t, value_t>;

    // Define the size of the input data for the first and second tables
    const uint64_t input_size_first = 10;
    const uint64_t input_size_second = 10;

    // Allocate host-sided (pinned) arrays for the input data of the first table
    key_t *keys_h_first; cudaMallocHost(&keys_h_first, sizeof(key_t) * input_size_first);
    value_t *values_h_first; cudaMallocHost(&values_h_first, sizeof(value_t) * input_size_first);

    // Generate data for the first table
    for (key_t i = 0; i < input_size_first; i++)
    {
        keys_h_first[i] = i + 1;
        values_h_first[i] = i + 2;
    }

    // Allocate host-sided (pinned) arrays for the input data of the second table
    key_t *keys_h_second; cudaMallocHost(&keys_h_second, sizeof(key_t) * input_size_second);
    value_t *values_h_second; cudaMallocHost(&values_h_second, sizeof(value_t) * input_size_second);

    // Generate data for the second table
    for (key_t i = 0; i < input_size_second; i++)
    {
        keys_h_second[i] = i + 5; // Common keys for join operation
        values_h_second[i] = i + 10;
    }

    // Allocate device-sided arrays for the input data of the first table
    key_t *keys_d_first; cudaMalloc(&keys_d_first, sizeof(key_t) * input_size_first);
    value_t *values_d_first; cudaMalloc(&values_d_first, sizeof(value_t) * input_size_first);

    // Allocate device-sided arrays for the input data of the second table
    key_t *keys_d_second; cudaMalloc(&keys_d_second, sizeof(key_t) * input_size_second);
    value_t *values_d_second; cudaMalloc(&values_d_second, sizeof(value_t) * input_size_second);

    // Copy input key/value pairs from the host to the device for the first table
    cudaMemcpy(keys_d_first, keys_h_first, sizeof(key_t) * input_size_first, cudaMemcpyHostToDevice);
    cudaMemcpy(values_d_first, values_h_first, sizeof(value_t) * input_size_first, cudaMemcpyHostToDevice);

    // Copy input key/value pairs from the host to the device for the second table
    cudaMemcpy(keys_d_second, keys_h_second, sizeof(key_t) * input_size_second, cudaMemcpyHostToDevice);
    cudaMemcpy(values_d_second, values_h_second, sizeof(value_t) * input_size_second, cudaMemcpyHostToDevice);

    // The target load factor of the hash table
    const float load = 0.9;

    // Calculate the capacity of the hash table
    const uint64_t capacity_first = input_size_first / load;

    // Initialize the hash table for the first table
    hash_table_t hash_table_first(capacity_first);

    // Insert data into the hash table for the first table
    hash_table_first.insert(keys_d_first, values_d_first, input_size_first);

    cudaDeviceSynchronize();

    // Perform hash join operation
    // Retrieve values from the hash table for the first table using keys from the second table
    value_t *result_d; cudaMalloc(&result_d, sizeof(value_t) * input_size_second);
    hash_table_first.retrieve(keys_d_second, input_size_second, result_d);

    // Allocate host-sided memory to copy the join result back to the host
    value_t* result_h;
    cudaMallocHost(&result_h, sizeof(value_t) * input_size_second);

    // Copy the result back to the host
    cudaMemcpy(result_h, result_d, sizeof(value_t) * input_size_second, cudaMemcpyDeviceToHost);

    // Print the join result
    for (uint64_t i = 0; i < input_size_second; i++) {
        std::cout << "Joined value for key " << keys_h_second[i] << ": " << result_h[i] << std::endl;
    }

    // Free allocated resources
    cudaFreeHost(result_h);

    // Free all allocated resources
    cudaFreeHost(keys_h_first);
    cudaFree(keys_d_first);
    cudaFreeHost(values_h_first);
    cudaFree(values_d_first);
    cudaFreeHost(keys_h_second);
    cudaFree(keys_d_second);
    cudaFreeHost(values_h_second);
    cudaFree(values_d_second);
    cudaFree(result_d);

    cudaDeviceSynchronize();

    return 0;
}
