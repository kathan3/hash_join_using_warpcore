#include <iostream>
#include <cstring>
#include <warpcore/single_value_hash_table.cuh>

int main()
{
    using key_t = std::uint32_t;
    using value_t = std::uint32_t;
    using char_t = char;
    using namespace warpcore;
    using hash_table_t = SingleValueHashTable<key_t, value_t>;

    const uint64_t input_size_first = 50;
    const uint64_t input_size_second = 50;
    const uint64_t num_value_columns_first = 4;
    const uint64_t num_char_columns_first = 3;
    const uint64_t num_value_columns_second = 5;
    const uint64_t num_char_columns_second = 3;

    // Allocate host-sided (pinned) arrays for the input data of the first table
    key_t *keys_h_first;
    value_t *values_h_first;
    value_t **value_columns_h_first = new value_t*[num_value_columns_first];
    char_t **char_columns_h_first = new char_t*[num_char_columns_first];
    cudaMallocHost(&keys_h_first, sizeof(key_t) * input_size_first);
    cudaMallocHost(&values_h_first, sizeof(value_t) * input_size_first);
    for (uint64_t col = 0; col < num_value_columns_first; ++col)
        cudaMallocHost(&value_columns_h_first[col], sizeof(value_t) * input_size_first);
    for (uint64_t col = 0; col < num_char_columns_first; ++col)
        cudaMallocHost(&char_columns_h_first[col], sizeof(char_t) * input_size_first);

    // Generate data for the first table
    for (key_t i = 0; i < input_size_first; i++)
    {
        keys_h_first[i] = i + 1;
        values_h_first[i] = i + 2;
        for (uint64_t col = 0; col < num_value_columns_first; ++col)
            value_columns_h_first[col][i] = i + col + 3;
        for (uint64_t col = 0; col < num_char_columns_first; ++col)
            char_columns_h_first[col][i] = 'A' + (i % 26);
    }

    // Allocate host-sided (pinned) arrays for the input data of the second table
    key_t *keys_h_second;
    value_t *values_h_second;
    value_t **value_columns_h_second = new value_t*[num_value_columns_second];
    char_t **char_columns_h_second = new char_t*[num_char_columns_second];
    cudaMallocHost(&keys_h_second, sizeof(key_t) * input_size_second);
    cudaMallocHost(&values_h_second, sizeof(value_t) * input_size_second);
    for (uint64_t col = 0; col < num_value_columns_second; ++col)
        cudaMallocHost(&value_columns_h_second[col], sizeof(value_t) * input_size_second);
    for (uint64_t col = 0; col < num_char_columns_second; ++col)
        cudaMallocHost(&char_columns_h_second[col], sizeof(char_t) * input_size_second);

    // Generate data for the second table
    for (key_t i = 0; i < input_size_second; i++)
    {
        keys_h_second[i] = i + 5; // Common keys for join operation
        values_h_second[i] = i + 10;
        for (uint64_t col = 0; col < num_value_columns_second; ++col)
            value_columns_h_second[col][i] = i + col + 11;
        for (uint64_t col = 0; col < num_char_columns_second; ++col)
            char_columns_h_second[col][i] = 'Z' - (i % 26);
    }

    // Allocate device-sided arrays for the input data of the first table
    key_t *keys_d_first;
    value_t *values_d_first;
    value_t **value_columns_d_first = new value_t*[num_value_columns_first];
    char_t **char_columns_d_first = new char_t*[num_char_columns_first];
    cudaMalloc(&keys_d_first, sizeof(key_t) * input_size_first);
    cudaMalloc(&values_d_first, sizeof(value_t) * input_size_first);
    for (uint64_t col = 0; col < num_value_columns_first; ++col)
        cudaMalloc(&value_columns_d_first[col], sizeof(value_t) * input_size_first);
    for (uint64_t col = 0; col < num_char_columns_first; ++col)
        cudaMalloc(&char_columns_d_first[col], sizeof(char_t) * input_size_first);

    // Allocate device-sided arrays for the input data of the second table
    key_t *keys_d_second;
    value_t *values_d_second;
    value_t **value_columns_d_second = new value_t*[num_value_columns_second];
    char_t **char_columns_d_second = new char_t*[num_char_columns_second];
    cudaMalloc(&keys_d_second, sizeof(key_t) * input_size_second);
    cudaMalloc(&values_d_second, sizeof(value_t) * input_size_second);
    for (uint64_t col = 0; col < num_value_columns_second; ++col)
        cudaMalloc(&value_columns_d_second[col], sizeof(value_t) * input_size_second);
    for (uint64_t col = 0; col < num_char_columns_second; ++col)
        cudaMalloc(&char_columns_d_second[col], sizeof(char_t) * input_size_second);

    // Copy input key/value pairs from the host to the device for the first table
    cudaMemcpy(keys_d_first, keys_h_first, sizeof(key_t) * input_size_first, cudaMemcpyHostToDevice);
    cudaMemcpy(values_d_first, values_h_first, sizeof(value_t) * input_size_first, cudaMemcpyHostToDevice);
    for (uint64_t col = 0; col < num_value_columns_first; ++col)
        cudaMemcpy(value_columns_d_first[col], value_columns_h_first[col], sizeof(value_t) * input_size_first, cudaMemcpyHostToDevice);
    for (uint64_t col = 0; col < num_char_columns_first; ++col)
        cudaMemcpy(char_columns_d_first[col], char_columns_h_first[col], sizeof(char_t) * input_size_first, cudaMemcpyHostToDevice);

    // Copy input key/value pairs from the host to the device for the second table
    cudaMemcpy(keys_d_second, keys_h_second, sizeof(key_t) * input_size_second, cudaMemcpyHostToDevice);
    cudaMemcpy(values_d_second, values_h_second, sizeof(value_t) * input_size_second, cudaMemcpyHostToDevice);
    for (uint64_t col = 0; col < num_value_columns_second; ++col)
        cudaMemcpy(value_columns_d_second[col], value_columns_h_second[col], sizeof(value_t) * input_size_second, cudaMemcpyHostToDevice);
    for (uint64_t col = 0; col < num_char_columns_second; ++col)
        cudaMemcpy(char_columns_d_second[col], char_columns_h_second[col], sizeof(char_t) * input_size_second, cudaMemcpyHostToDevice);

    // Calculate the capacity of the hash table
    const float load = 0.9;
    const uint64_t capacity_first = input_size_first / load;

    // Initialize the hash table for the first table
    hash_table_t hash_table_first(capacity_first);

    // Insert data into the hash table for the first table
    hash_table_first.insert(keys_d_first, values_d_first, input_size_first);
    cudaDeviceSynchronize();

    // Perform hash join operation
    value_t *result_d;
    cudaMalloc(&result_d, sizeof(value_t) * input_size_second);
    hash_table_first.retrieve(keys_d_second, input_size_second, result_d);

    // Allocate host-sided memory to copy the join result back to the host
    value_t* result_h;
    cudaMallocHost(&result_h, sizeof(value_t) * input_size_second);
    cudaMemcpy(result_h, result_d, sizeof(value_t) * input_size_second, cudaMemcpyDeviceToHost);

    // Print the join result
    for (uint64_t i = 0; i < input_size_second; i++) {
        std::cout << "Key: " << keys_h_second[i] << " -> ";
        std::cout << "First Table Columns: ";
        for (uint64_t col = 0; col < num_value_columns_first; ++col)
            std::cout << value_columns_h_first[col][result_h[i] - 1] << " ";
        for (uint64_t col = 0; col < num_char_columns_first; ++col)
            std::cout << char_columns_h_first[col][result_h[i] - 1] << " ";
        std::cout << "| Second Table Columns: ";
        for (uint64_t col = 0; col < num_value_columns_second; ++col)
            std::cout << value_columns_h_second[col][i] << " ";
        for (uint64_t col = 0; col < num_char_columns_second; ++col)
            std::cout << char_columns_h_second[col][i] << " ";
        std::cout << std::endl;
    }

    // Free allocated resources
    cudaFreeHost(result_h);
    for (uint64_t col = 0; col < num_value_columns_first; ++col)
        cudaFreeHost(value_columns_h_first[col]);
    for (uint64_t col = 0; col < num_char_columns_first; ++col)
        cudaFreeHost(char_columns_h_first[col]);
    delete[] value_columns_h_first;
    delete[] char_columns_h_first;

    cudaFreeHost(keys_h_first);
    cudaFree(keys_d_first);
    cudaFreeHost(values_h_first);
    cudaFree(values_d_first);
    for (uint64_t col = 0; col < num_value_columns_first; ++col)
        cudaFree(value_columns_d_first[col]);
    for (uint64_t col = 0; col < num_char_columns_first; ++col)
        cudaFree(char_columns_d_first[col]);
    delete[] value_columns_d_first;
    delete[] char_columns_d_first;

    cudaFreeHost(keys_h_second);
    cudaFree(keys_d_second);
    cudaFreeHost(values_h_second);
    cudaFree(values_d_second);
    for (uint64_t col = 0; col < num_value_columns_second; ++col)
        cudaFree(value_columns_d_second[col]);
    for (uint64_t col = 0; col < num_char_columns_second; ++col)
        cudaFree(char_columns_d_second[col]);
    delete[] value_columns_d_second;
    delete[] char_columns_d_second;

    cudaFree(result_d);
    cudaDeviceSynchronize();

    return 0;
}
