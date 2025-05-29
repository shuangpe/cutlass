#!/bin/bash

BUILD_DIR="build"

# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
# sh cuda_12.8.0_570.86.10_linux.run
# echo "export PATH=/usr/local/cuda-12.8/bin:$PATH" >> ~/.bashrc
# echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
# source ~/.bashrc

# rm -rf $BUILD_DIR
cmake -S . -B $BUILD_DIR -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_NVCC_KEEP=ON -DCMAKE_CXX_FLAGS="-D__STRICT_ANSI__"

# Initialize an empty list to store filenames
file_list=()
cubin_list=()

# Function to recursively traverse directories
traverse_directory() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively traverse it
            traverse_directory "$file"
        elif [ -f "$file" ] && [[ "$file" == *.cu ]]; then
            # If it's a .cu file, check if the path contains "blackwell"
            if [[ "$file" == *blackwell* ]]; then
                # Extract the filename without the extension
                filename=$(basename "$file" .cu)

                if [ "$filename" == "blackwell_gemm_preferred_cluster" ]; then
                    file_list+=("73_$filename")
                elif [ "$filename" == "blackwell_gemm_streamk" ]; then
                    file_list+=("74_$filename")
                elif [[ "$filename" == 77_blackwell_fmha* ]]; then
                    file_list+=("${filename}_fp8" "${filename}_fp16")
                else
                    file_list+=("$filename")
                fi

            fi
        fi
    done
}

# Function to recursively traverse directories and find .cubin files
find_cubin_files() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively traverse it
            find_cubin_files "$file"
        elif [ -f "$file" ] && [[ "$file" == *.cubin ]]; then
            # Add the .cubin file to the list
            cubin_list+=("$file")
        fi
    done
}

# Start traversing from the ./examples directory
traverse_directory "./examples"

# Iterate over the list and execute make for each filename
for filename in "${file_list[@]}"; do
    # Skip if filename doesn't contain 72
    if [[ "$filename" != *72* ]]; then
        echo "Skipping $filename (does not contain 72)"
        continue
    fi
    echo "cmake --build $BUILD_DIR --target $filename"
    cmake --build $BUILD_DIR --target $filename
done

# Start traversing from the current build directory to find .cubin files
find_cubin_files "$BUILD_DIR/examples"

# Iterate over the list and execute cuobjdump --dump-sass for each .cubin file
for cubin_file in "${cubin_list[@]}"; do
    # Determine the output .sass file path
    sass_file="${cubin_file%.cubin}.sass"
    # Execute cuobjdump and redirect the output to the .sass file
    echo "cuobjdump --dump-sass $cubin_file > $sass_file"
    cuobjdump --dump-sass $cubin_file > $sass_file
done

# Create the cutlass_sm100a_ptx_sass directory if it doesn't exist
output_dir="cutlass_sm100a_ptx_sass"
rm -rf $output_dir $output_dir.zip; mkdir -p $output_dir

# Function to recursively find and copy .ptx and .sass files
copy_ptx_sass_files() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively traverse it
            copy_ptx_sass_files "$file"
        elif [ -f "$file" ] && [[ "$file" == *blackwell* ]] && ([[ "$file" == *.ptx ]] || [[ "$file" == *.sass ]]); then
            # Copy the .ptx and .sass files to the output directory
            echo "cp $file $output_dir"
            cp $file $output_dir
        fi
    done
}

# Start traversing from the current build directory to find and copy .ptx and .sass files
copy_ptx_sass_files "$BUILD_DIR/examples"

zip -r $output_dir.zip $output_dir

echo "All .ptx and .sass files have been copied to $output_dir"
