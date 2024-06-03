#include <iostream>
#include <vector>
#include <string>
#include <warpcore/multi_value_hash_table.cuh>
#include <cuda_runtime.h>

struct Student {
    uint32_t sr_no;
    char name[32];
};

struct Grade {
    uint32_t sr_no;
    uint32_t grade;
};

int main()
{
    using namespace warpcore;

    // key/value types
    using key_t = std::uint32_t;
    using student_value_t = Student;
    using grade_value_t = Grade;

    // configure the hash tables
    using student_hash_table_t = MultiValueHashTable<
        key_t,
        student_value_t,
        defaults::empty_key<key_t>(), // empty sentinel
        defaults::tombstone_key<key_t>(), // tombstone sentinel
        defaults::probing_scheme_t<key_t, 8>>; // the cooperative probing scheme

    using grade_hash_table_t = MultiValueHashTable<
        key_t,
        grade_value_t,
        defaults::empty_key<key_t>(), // empty sentinel
        defaults::tombstone_key<key_t>(), // tombstone sentinel
        defaults::probing_scheme_t<key_t, 8>>; // the cooperative probing scheme

    // this type represents the current status (errors/warnings) of the tables
    using status_t = typename student_hash_table_t::status_type;
    // we want to catch the status per-query -> configure status handle
    using status_handler_t = typename status_handlers::ReturnStatus;

    // number of unique keys
    const index_t size_unique_keys = 5;
    // the actual number of input key/value pairs
    const index_t student_size = 5;
    const index_t grade_size = 4;
    // target load factor of the hash tables
    const float load_factor = 0.8;

    // Initialize the hash tables
    student_hash_table_t student_hash_table(student_size / load_factor);
    grade_hash_table_t grade_hash_table(grade_size / load_factor);
    cudaDeviceSynchronize(); CUERR

    // Allocate host and device memory for students
    key_t *student_keys_h = nullptr;
    cudaMallocHost(&student_keys_h, sizeof(key_t) * student_size); CUERR
    key_t *student_keys_d = nullptr;
    cudaMalloc(&student_keys_d, sizeof(key_t) * student_size); CUERR

    student_value_t *student_values_h = nullptr;
    cudaMallocHost(&student_values_h, sizeof(student_value_t) * student_size); CUERR
    student_value_t *student_values_d = nullptr;
    cudaMalloc(&student_values_d, sizeof(student_value_t) * student_size); CUERR

    // Allocate host and device memory for grades
    key_t *grade_keys_h = nullptr;
    cudaMallocHost(&grade_keys_h, sizeof(key_t) * grade_size); CUERR
    key_t *grade_keys_d = nullptr;
    cudaMalloc(&grade_keys_d, sizeof(key_t) * grade_size); CUERR

    grade_value_t *grade_values_h = nullptr;
    cudaMallocHost(&grade_values_h, sizeof(grade_value_t) * grade_size); CUERR
    grade_value_t *grade_values_d = nullptr;
    cudaMalloc(&grade_values_d, sizeof(grade_value_t) * grade_size); CUERR

    // Output buffers
    index_t *offsets_out_d = nullptr;
    cudaMalloc(&offsets_out_d, sizeof(index_t) * (student_size + 1)); CUERR

    student_value_t *student_values_out_d = nullptr;
    cudaMalloc(&student_values_out_d, sizeof(student_value_t) * student_size); CUERR
    grade_value_t *grade_values_out_d = nullptr;
    cudaMalloc(&grade_values_out_d, sizeof(grade_value_t) * student_size); CUERR

    status_t *status_h = nullptr;
    cudaMallocHost(&status_h, sizeof(status_t) * student_size); CUERR
    status_t *status_d = nullptr;
    cudaMalloc(&status_d, sizeof(status_t) * student_size); CUERR

    // Initialize students
    std::vector<Student> students = {
        {1, "Alice"},
        {2, "Bob"},
        {3, "Charlie"},
        {4, "David"},
        {5, "Eve"}
    };

    // Initialize grades
    std::vector<Grade> grades = {
        {2, 85},
        {3, 90},
        {4, 75},
        {5, 95}
    };

    #pragma omp parallel for
    for (index_t i = 0; i < student_size; ++i)
    {
        student_keys_h[i] = students[i].sr_no;
        snprintf(student_values_h[i].name, sizeof(student_values_h[i].name), "%s", students[i].name);
        student_values_h[i].sr_no = students[i].sr_no;

        status_h[i] = status_t::none();
    }

    #pragma omp parallel for
    for (index_t i = 0; i < grade_size; ++i)
    {
        grade_keys_h[i] = grades[i].sr_no;
        grade_values_h[i].sr_no = grades[i].sr_no;
        grade_values_h[i].grade = grades[i].grade;
    }

    // Copy student data from host to device
    cudaMemcpy(student_keys_d, student_keys_h, sizeof(key_t) * student_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(student_values_d, student_values_h, sizeof(student_value_t) * student_size, cudaMemcpyHostToDevice); CUERR

    // Copy grade data from host to device
    cudaMemcpy(grade_keys_d, grade_keys_h, sizeof(key_t) * grade_size, cudaMemcpyHostToDevice); CUERR
    cudaMemcpy(grade_values_d, grade_values_h, sizeof(grade_value_t) * grade_size, cudaMemcpyHostToDevice); CUERR

    // Insert student key/value pairs into the student hash table
    student_hash_table.insert<status_handler_t>(
        student_keys_d,
        student_values_d,
        student_size,
        0,
        defaults::probing_length(),
        status_d);
    cudaDeviceSynchronize(); CUERR

    // Insert grade key/value pairs into the grade hash table
    grade_hash_table.insert<status_handler_t>(
        grade_keys_d,
        grade_values_d,
        grade_size,
        0,
        defaults::probing_length(),
        status_d);
    cudaDeviceSynchronize(); CUERR

    // Retrieve values from student hash table
    index_t student_value_size = 0;
    student_hash_table.retrieve<status_handler_t>(
        student_keys_d,
        student_size,
        offsets_out_d,
        offsets_out_d + 1,
        student_values_out_d,
        student_value_size,
        0,
        defaults::probing_length(),
        status_d);
    cudaDeviceSynchronize(); CUERR

    // Retrieve values from grade hash table
    index_t grade_value_size = 0;
    grade_hash_table.retrieve<status_handler_t>(
        grade_keys_d,
        grade_size,
        offsets_out_d,
        offsets_out_d + 1,
        grade_values_out_d,
        grade_value_size,
        0,
        defaults::probing_length(),
        status_d);
    cudaDeviceSynchronize(); CUERR

    // Allocate host vectors for results
    std::vector<key_t> student_keys_out_h(student_size);
    std::vector<student_value_t> student_values_out_h(student_size);
    std::vector<index_t> student_offsets_out_h(student_size + 1);

    std::vector<key_t> grade_keys_out_h(grade_size);
    std::vector<grade_value_t> grade_values_out_h(grade_size);
    std::vector<index_t> grade_offsets_out_h(grade_size + 1);

    // Copy data from device to host
    cudaMemcpy(student_keys_out_h.data(), student_keys_d, sizeof(key_t) * student_size, cudaMemcpyDeviceToHost); CUERR
    cudaMemcpy(student_values_out_h.data(), student_values_out_d, sizeof(student_value_t) * student_size, cudaMemcpyDeviceToHost); CUERR
    cudaMemcpy(student_offsets_out_h.data(), offsets_out_d, sizeof(index_t) * (student_size + 1), cudaMemcpyDeviceToHost); CUERR

    cudaMemcpy(grade_keys_out_h.data(), grade_keys_d, sizeof(key_t) * grade_size, cudaMemcpyDeviceToHost); CUERR
    cudaMemcpy(grade_values_out_h.data(), grade_values_out_d, sizeof(grade_value_t) * grade_size, cudaMemcpyDeviceToHost); CUERR
    cudaMemcpy(grade_offsets_out_h.data(), offsets_out_d, sizeof(index_t) * (grade_size + 1), cudaMemcpyDeviceToHost); CUERR

    // Print joined (sr_no, name, grade) pairs
    std::cout << "SR_NO\tName\tGrade" << std::endl;
    for (index_t i = 0; i < student_size; ++i)
    {
        std::cout << student_values_out_h[i].sr_no << "\t" << student_values_out_h[i].name << "\t";
        bool found = false;
        for (index_t j = 0; j < grade_size; ++j)
        {
            if (student_values_out_h[i].sr_no == grade_values_out_h[j].sr_no)
            {
                std::cout << grade_values_out_h[j].grade;
                found = true;
                break;
            }
        }
        if (!found) std::cout << "N/A";
        std::cout << std::endl;
    }

    // Free memory
    cudaFreeHost(student_keys_h);
    cudaFreeHost(student_values_h);
    cudaFreeHost(status_h);
    cudaFree(student_keys_d);
    cudaFree(student_values_d);
    cudaFree(student_values_out_d);
    cudaFree(offsets_out_d);
    cudaFree(status_d);

    cudaFreeHost(grade_keys_h);
    cudaFreeHost(grade_values_h);
    cudaFree(grade_keys_d);
    cudaFree(grade_values_d);

    cudaDeviceSynchronize(); CUERR
}
