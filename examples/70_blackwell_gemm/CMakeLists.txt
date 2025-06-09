
# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


set(TEST_SWIZZLE_1 --swizzle=1)
set(TEST_SWIZZLE_2 --swizzle=2)
set(TEST_SWIZZLE_5 --swizzle=5)
set(TEST_SWIZZLE_5_UNEVEN --swizzle=5 --m=4096 --n=16384)

if(NOT CUTLASS_NVCC_ARCHS STREQUAL "100")
cutlass_example_add_executable(
  70_blackwell_fp16_gemm_256x256x128_2x1x1
  70_blackwell_fp16_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_SWIZZLE_1
  TEST_SWIZZLE_2
  TEST_SWIZZLE_5
  TEST_SWIZZLE_5_UNEVEN
)
target_compile_definitions(70_blackwell_fp16_gemm_256x256x128_2x1x1 PRIVATE MS_M=_256 MS_K=_128 CS_M=_2 CS_N=_1)

cutlass_example_add_executable(
  70_blackwell_fp16_gemm_256x256x128_2x1x1_hacking
  70_blackwell_fp16_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_SWIZZLE_1
  TEST_SWIZZLE_2
  TEST_SWIZZLE_5
  TEST_SWIZZLE_5_UNEVEN
)
target_compile_definitions(70_blackwell_fp16_gemm_256x256x128_2x1x1_hacking PRIVATE MS_M=_256 MS_K=_128 CS_M=_2 CS_N=_1)
target_compile_definitions(70_blackwell_fp16_gemm_256x256x128_2x1x1_hacking PRIVATE CUTLASS_UNIT_TEST_PIPELINE=1 HACK_GEMM_WRITE_SLM_ONCE=1)

cutlass_example_add_executable(
  70_blackwell_fp16_gemm_128x256x128_1x1x1
  70_blackwell_fp16_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_SWIZZLE_1
  TEST_SWIZZLE_2
  TEST_SWIZZLE_5
  TEST_SWIZZLE_5_UNEVEN
)
target_compile_definitions(70_blackwell_fp16_gemm_128x256x128_1x1x1 PRIVATE MS_M=_128 MS_K=_64 CS_M=_1 CS_N=_1)

cutlass_example_add_executable(
  70_blackwell_fp8_gemm
  70_blackwell_fp8_gemm.cu
  TEST_COMMAND_OPTIONS
  TEST_SWIZZLE_1
  TEST_SWIZZLE_2
  TEST_SWIZZLE_5
  TEST_SWIZZLE_5_UNEVEN
)
endif()
