#!/bin/bash

BUILD_DIR="build"
BASE_DIR="./test/unit"

cmake_targets=()

# Define a function to extract CMake targets
build_sm100_ut() {
    local file=$1
    local matches

    matches=$(python3 - <<END
import re

with open("$file", "r") as file:
    content = file.read()

pattern = re.compile(r'cutlass_test_unit_gemm_device_add_executable\(\s*(\S+)', re.MULTILINE)
matches = pattern.findall(content)

for match in matches:
    if "sm100" in match:
        print(match)
END
)

    for match in $matches; do
        cmake_targets+=("$match")
    done

    for item in "${cmake_targets[@]}"; do
        echo "cmake --build $BUILD_DIR --target $item"
        cmake --build $BUILD_DIR --target $item
    done
}

dump_sass() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            dump_sass "$file"
        elif [ -f "$file" ] && [[ "$file" == *.cubin ]]; then
            sass_file="${file%.cubin}.sass"
            echo "cuobjdump --dump-sass $file > $sass_file"
            cuobjdump --dump-sass $file > $sass_file
        fi
    done
}

output_dir="cutlass_sm100a_ut_ptx_sass"
rm -rf $output_dir $output_dir.zip; mkdir -p $output_dir

copy_ptx_sass() {
    local dir="$1"
    for file in "$dir"/*; do
        if [ -d "$file" ]; then
            # If it's a directory, recursively traverse it
            copy_ptx_sass "$file"
        elif [ -f "$file" ] && [[ "$file" == *sm100* ]] && ([[ "$file" == *.ptx ]] || [[ "$file" == *.sass ]]); then
            # Copy the .ptx and .sass files to the output directory
            echo "cp $file $output_dir"
            cp $file $output_dir
        fi
    done
}

Traverse all CMakeLists.txt files under the test/unit directory
find $BASE_DIR -name "CMakeLists.txt" | while read -r cmake_file; do
    build_sm100_ut $cmake_file
done

dump_sass "$BUILD_DIR/test/unit"
copy_ptx_sass "$BUILD_DIR/test/unit"

zip -r $output_dir.zip $output_dir

echo "All .ptx and .sass files have been copied to $output_dir"
