/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief A FP16 dense GEMM example for the NVIDIA Blackwell SM100 architecture using CUTLASS.

    This example demonstrates minimal set of changes needed to transition from a Hopper CUTLASS 3.x 
    GEMM kernel (see example 48_hopper_warp_specialized_gemm) to a Blackwell 3.x CUTLASS GEMM kernel.
    
    The Blackwell SM100 CUTLASS kernel uses of the following Blackwell SM100 features:

    1. New series of Tensor Core MMA Instructions (tcgen05) introduced on the Blackwell architecture (sm100a) 
    which have 2x throughput compared to Hopper Tensor Core MMA instructions (WGMMA). 
    
    Note that Hopper WGMMA Tensor Core MMA instructions are not compatible on Blackwell (See https://docs.nvidia.com/cuda/parallel-thread-execution). 

    2. A new per-SM memory called Tensor Memory (TMEM) introduced on the Blackwell architecture (sm100a). 
    Blackwell SM100 Tensor Core MMA instructions store their accumulation results in TMEM instead of the 
    Register File. (Please refer to CUDA 12.8 docs on https://docs.nvidia.com/cuda/).

    3. An extended flavor of the warp-specialized kernel design introduced in Hopper enabled by use of TMEM 
    which allows us to decouple the execution of MMA and epilogue into separate warps. 

    4. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution). 

    Usage:
      $ ./examples/70_blackwell_gemm/70_blackwell_fp16_gemm --m=8192 --n=8192 --k=8192
*/



#include <iostream>
#include <chrono>
#include <numeric>
#include <random>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = half_t;                                         // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = half_t;                                         // Element type for B matrix operand
using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = half_t;                                          // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag

// MMA and Cluster Tile Shapes
// Shape of the tile computed by tcgen05 MMA, could be across 2 SMs if Cluster Shape %2 == 0 
using MmaTileShape_MNK = Shape<_256,_256,_64>;
// Shape of the threadblocks in a cluster
using ClusterShape_MNK = Shape<_2,_1,_1>;

// Build the epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, 
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Build the mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

// Compose into a kernel
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int, int>, // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;                   // Default to ClusterLaunchControl (CLC) based tile scheduler 

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle;
  int mask_ratio;

  Options():
    help(false),
    m(16384), n(16384), k(16384),
    alpha(1.f), beta(0.f),
    iterations(2000),
    swizzle(0),
    mask_ratio(0)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("swizzle", swizzle);
    cmd.get_cmd_line_argument("mask_ratio", mask_ratio, 0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "70_blackwell_fp16_gemm\n\n"
      << "  Blackwell FP16 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n\n"
        << "  --mask_ratio=<int>           Percentage of elements to mask (set to zero) in A and B, default 0 (no mask)\n\n"
        << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "70_blackwell_fp16_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(8);
    scope_min = Element(-8);
  }

  scope_max = Element(5);
  scope_min = Element(-5);

  cutlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);

  std::vector<typename Gemm::ElementA> host_A(options.m * options.k);
  std::vector<typename Gemm::ElementB> host_B(options.k * options.n);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());
  block_A.copy_to_host(host_A.data(), host_A.size());
  block_B.copy_to_host(host_B.data(), host_B.size());

  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using CtaShape_MNK = typename CollectiveMainloop::CtaShape_MNK;

  int tile_m = size<0>(CtaShape_MNK{});
  int tile_n = size<1>(CtaShape_MNK{});
  int tile_k = size<2>(CtaShape_MNK{});

  std::cout << "Stages: " << DispatchPolicy::Stages << std::endl;
  std::cout << "TileShape: " << tile_m << "x" << tile_n << "x" << tile_k << std::endl;

#if HACK_GEMM_WRITE_SLM_ONCE
  std::cout << "HackLoadG2L: 1" << std::endl;
#else
  std::cout << "HackLoadG2L: 0" << std::endl;
#endif

  // Mask host_A and host_B if mask_ratio > 0
  if (options.mask_ratio > 0) {
    float mask_ratio_f = options.mask_ratio / 100.0f;

    // Only mask the first tile
    size_t first_tile_size_A = tile_m * tile_k;
    size_t first_tile_size_B = tile_k * tile_n;

    size_t num_mask_A = static_cast<size_t>(first_tile_size_A * mask_ratio_f);
    size_t num_mask_B = static_cast<size_t>(first_tile_size_B * mask_ratio_f);

    std::vector<size_t> indices_A(first_tile_size_A);
    std::vector<size_t> indices_B(first_tile_size_B);

    std::iota(indices_A.begin(), indices_A.end(), 0);
    std::iota(indices_B.begin(), indices_B.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_A.begin(), indices_A.end(), g);
    std::shuffle(indices_B.begin(), indices_B.end(), g);

    // Mask elements in the first tile of A
    for (size_t i = 0; i < num_mask_A; ++i) {
      size_t idx = indices_A[i];
      int m = idx / tile_k;
      int k = idx % tile_k;
      host_A[m * options.k + k] = typename Gemm::ElementA(0);
    }

    // Mask elements in the first tile of B
    for (size_t i = 0; i < num_mask_B; ++i) {
      size_t idx = indices_B[i];
      int k = idx / tile_n;
      int n = idx % tile_n;
      host_B[k * options.n + n] = typename Gemm::ElementB(0);
    }
  }

  for (int m = 0; m < options.m; ++m) {
    for (int k = 0; k < options.k; ++k) {
      if (m < tile_m && k < tile_k) continue;
      host_A[m * options.k + k] = host_A[(m % tile_m) * options.k + (k % tile_k)];
    }
  }

  for (int k = 0; k < options.k; ++k) {
    for (int n = 0; n < options.n; ++n) {
      if (k < tile_k && n < tile_n) continue;
      host_B[k * options.n + n] = host_B[(k % tile_k) * options.n + (n % tile_n)];
    }
  }

  if (std::is_same<LayoutB, cutlass::layout::ColumnMajor>::value) {
    std::vector<typename Gemm::ElementB> temp(options.k * options.n);
    temp = host_B; // Copy the original data

    for (int k = 0; k < options.k; ++k) {
      for (int n = 0; n < options.n; ++n) {
        host_B[n * options.k + k] = temp[k * options.n + n];
      }
    }
  }

  block_A.copy_from_host(host_A.data());
  block_B.copy_from_host(host_B.data());
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D}
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());

  if (!passed) {
    int mismatch_count = 0;
    std::vector<typename Gemm::ElementD> host_D(options.m*options.n);
    std::vector<typename Gemm::ElementD> host_ref_D(options.m*options.n);
    block_D.copy_to_host(host_D.data(), host_D.size());
    block_ref_D.copy_to_host(host_ref_D.data(), host_ref_D.size());

    for (int m = 0; m < options.m; ++m) {
      for (int n = 0; n < options.n; ++n) {
        if (host_D[m * options.n + n] != host_ref_D[m * options.n + n]) {
          std::cout << "Mismatch at (" << m << ", " << n << "): "
                    << "D = " << static_cast<float>(host_D[m * options.n + n])
                    << ", Ref D = " << static_cast<float>(host_ref_D[m * options.n + n])
                    << std::endl;
          break; // Exit inner loop on first mismatch
        }
      }
      if (mismatch_count > 8) break; // Exit outer loop if mismatch found
    }
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  auto grid_shape = Gemm::get_grid_shape(gemm.params());
  std::cout << "GridDims: " << grid_shape.x << "x" << grid_shape.y << "x" << grid_shape.z << std::endl;

  auto get_timestamp = []() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << now_ms.count();
    return timestamp.str();
  };

  std::cout << "  [" << get_timestamp() << "] Start warmup and correctness check for CUTLASS kernel." << std::endl;

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  [" << get_timestamp() << "]";
  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
     std::cout << "  [" << get_timestamp() << "] "
               << "Start profiling CUTLASS kernel for " << options.iterations << " iterations." << std::endl;
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      CUTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  [" << get_timestamp() << "] "
              << "Profiling completed. Results:" << std::endl;
    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  MaskRatio: " << options.mask_ratio << "%" << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
    std::cout << "  TFLOPS: " << result.gflops / 1000.0 << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 100a.

  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));  
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 10 || props.minor != 0) {
    std::cerr << "This example requires a GPU with compute capability 100a)." << std::endl;
    return 0;
  } 
  
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  run<Gemm>(options);
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
