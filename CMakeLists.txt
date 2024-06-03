cmake_minimum_required(VERSION 3.18)
project(hash_join_warpcore LANGUAGES CXX CUDA)
 
# Include updated CPM.cmake for CPMAddPackage function
include(cmake/CPM.cmake)
 
# Add Warpcore as a dependency using CPM
CPMAddPackage(
    NAME warpcore
    GITHUB_REPOSITORY sleeepyjack/warpcore
    GIT_TAG 1.0.0-alpha.1 # Replace with an existing tag/version of Warpcore
)
 
# Define your target
add_executable(hash_join_warpcore src/hash_join_static_map.cu)

# Link cuco to your target
target_link_libraries(hash_join_warpcore PRIVATE warpcore)

# Specify CUDA properties
set_target_properties(hash_join_warpcore PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "61"  # Set appropriate architectures for your hardware
)

 #Add required compiler flags
target_compile_options(hash_join_warpcore PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
    >
)
