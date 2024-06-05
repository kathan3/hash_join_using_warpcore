#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <warpcore/single_value_hash_table.cuh>

// Simple Timer class for measuring durations
class Timer {
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double getDuration() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
};

// Helper function to read CSV file and extract all columns and specified key column
void read_csv(const std::string& filename, std::vector<std::vector<std::string>>& table_data, std::vector<uint32_t>& keys, std::vector<uint32_t>& row_identifiers, int key_col_index) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    uint32_t row_id = 0;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(ss, cell, '|')) {
            row.push_back(cell);
        }

        if (row.size() > key_col_index) {
            keys.push_back(std::stoul(row[key_col_index]));
            row_identifiers.push_back(row_id);
            table_data.push_back(row);
        } else {
            std::cerr << "Warning: Not enough columns in row " << row_id << std::endl;
        }

        row_id++;
    }

    file.close();
}

// Function to get the current timestamp as a string
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S");
    return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <file1> <key_col1> <file2> <key_col2>" << std::endl;
        return 1;
    }

    std::string file_first = argv[1];
    int key_col_first = std::stoi(argv[2]);
    std::string file_second = argv[3];
    int key_col_second = std::stoi(argv[4]);

    std::string output_file = "join_output_" + get_timestamp() + ".txt";

     // Declare timers for measuring each operation
    Timer transfer_time, hash_table_time, probing_time, result_transfer_time, output_fetch_time;

    using key_t = std::uint32_t;
    using value_t = std::uint32_t;
    using namespace warpcore;
    using hash_table_t = SingleValueHashTable<key_t, value_t>;

    std::vector<std::vector<std::string>> table_data_first;
    std::vector<key_t> keys_h_first;
    std::vector<value_t> row_identifiers_h_first;
    std::vector<std::vector<std::string>> table_data_second;
    std::vector<key_t> keys_h_second;
    std::vector<value_t> row_identifiers_h_second;

    read_csv(file_first, table_data_first, keys_h_first, row_identifiers_h_first, key_col_first);
    read_csv(file_second, table_data_second, keys_h_second, row_identifiers_h_second, key_col_second);

    const uint64_t input_size_first = keys_h_first.size();
    const uint64_t input_size_second = keys_h_second.size();
    if (input_size_first == 0 || input_size_second == 0) {
        std::cerr << "Error: One of the input tables is empty" << std::endl;
        return 1;
    }

    transfer_time.start();
    key_t *keys_d_first, *keys_d_second;
    value_t *row_identifiers_d_first, *row_identifiers_d_second;
    cudaError_t err;

    err = cudaMalloc(&keys_d_first, sizeof(key_t) * input_size_first);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc keys_d_first failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMalloc(&row_identifiers_d_first, sizeof(value_t) * input_size_first);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc row_identifiers_d_first failed with " << cudaGetErrorString(err) << std::endl;
        cudaFree(keys_d_first);
        return 1;
    }

    err = cudaMalloc(&keys_d_second, sizeof(key_t) * input_size_second);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc keys_d_second failed with " << cudaGetErrorString(err) << std::endl;
        cudaFree(keys_d_first);
        cudaFree(row_identifiers_d_first);
        return 1;
    }

    err = cudaMalloc(&row_identifiers_d_second, sizeof(value_t) * input_size_second);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc row_identifiers_d_second failed with " << cudaGetErrorString(err) << std::endl;
        cudaFree(keys_d_first);
        cudaFree(row_identifiers_d_first);
        cudaFree(keys_d_second);
        return 1;
    }

    err = cudaMemcpy(keys_d_first, keys_h_first.data(), sizeof(key_t) * input_size_first, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy keys_d_first failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy(row_identifiers_d_first, row_identifiers_h_first.data(), sizeof(value_t) * input_size_first, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy row_identifiers_d_first failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy(keys_d_second, keys_h_second.data(), sizeof(key_t) * input_size_second, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy keys_d_second failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy(row_identifiers_d_second, row_identifiers_h_second.data(), sizeof(value_t) * input_size_second, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy row_identifiers_d_second failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    transfer_time.stop();

    hash_table_time.start();
    const float load = 0.9;
    const uint64_t capacity_first = input_size_first / load;

    hash_table_t hash_table_first(capacity_first);

    hash_table_first.insert(keys_d_first, row_identifiers_d_first, input_size_first);
    cudaDeviceSynchronize();
    hash_table_time.stop();

    probing_time.start();
    value_t *result_d;
    err = cudaMalloc(&result_d, sizeof(value_t) * input_size_second);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMalloc result_d failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    hash_table_first.retrieve(keys_d_second, input_size_second, result_d);

    probing_time.stop();

    result_transfer_time.start();
    value_t* result_h = new value_t[input_size_second];
    err = cudaMemcpy(result_h, result_d, sizeof(value_t) * input_size_second, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error: cudaMemcpy result_h failed with " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    result_transfer_time.stop();

    output_fetch_time.start();
    std::ofstream output(output_file);
    if (!output.is_open()) {
        std::cerr << "Error: Could not open output file " << output_file << std::endl;
        return 1;
    }

    for (uint64_t i = 0; i < input_size_second; i++) {
        if (result_h[i] != static_cast<value_t>(-1)) { // Assuming -1 denotes no match
            for (const auto& cell : table_data_first[result_h[i]]) {
                output << cell << "|";
            }
            output << " || ";
            for (const auto& cell : table_data_second[row_identifiers_h_second[i]]) {
                output << cell << "|";
            }
            output << std::endl;
        }
    }
     output_fetch_time.stop();

    delete[] result_h;
    cudaFree(keys_d_first);
    cudaFree(row_identifiers_d_first);
    cudaFree(keys_d_second);
    cudaFree(row_identifiers_d_second);
    cudaFree(result_d);
    cudaDeviceSynchronize();

    output.close();
    // Print the time for each operation
    std::cout << "Time for transferring keys and data from host to device for both the table: " << transfer_time.getDuration() << " ms" << std::endl;
    std::cout << "Time for hash table creation and insertion for table 1: " << hash_table_time.getDuration() << " ms" << std::endl;
    std::cout << "Time for probing the hash table: " << probing_time.getDuration() << " ms" << std::endl;
    std::cout << "Time for transferring the join result from device to host: " << result_transfer_time.getDuration() << " ms" << std::endl;
    std::cout << "Time for fetching final output from both tables: " << output_fetch_time.getDuration() << " ms" << std::endl;
    std::cout << "Total time for hash-join: " << transfer_time.getDuration()+ hash_table_time.getDuration()+probing_time.getDuration()+result_transfer_time.getDuration()+output_fetch_time.getDuration() << " ms" << std::endl;

    std::cout << "Join results written to " << output_file << std::endl;
    return 0;
}
