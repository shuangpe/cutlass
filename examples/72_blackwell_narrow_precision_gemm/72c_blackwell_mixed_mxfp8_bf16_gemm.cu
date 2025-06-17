/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief A GEMM example using CUTLASS for the NVIDIA Blackwell SM100 architecture.

    This example demonstrates a simple way to instantiate and run a mixed precision blockscaled GEMM on the NVIDIA Blackwell SM100 architecture.
    This Blackwell SM100 CUTLASS kernel uses the new Block Scaled Tensor Core MMA Instructions (tcgen05.mma.blockscaled) introduced
    on the Blackwell architecture (sm100a) which have the same throughput compared to fp8 Tensor Core MMA instructions (tcgen05.mma)
    and 2x throughput compared to fp8 Hopper Tensor Core MMA Instructions (WGMMA) (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Similar to 72a_blackwell_nvfp4_fp32_gemm, this kernel leverages:
    1. Blockscaled tcgen05.mma instructions.

    2. Per-SM memory called Tensor Memory (TMEM) (Please refer to CUDA 12.8 docs on https://docs.nvidia.com/cuda/).
    
    3. The extended warp-specialized kernel design introduced in Hopper enabled by use of TMEM 
    which allows us to decouple the execution of MMA and epilogue into separate warps. 
    
    4. A new SW controlled dynamic scheduler based on cluster launch control (See https://docs.nvidia.com/cuda/parallel-thread-execution).

    Usage:

      $ ./examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm --m=2048 --n=2048 --k=2048
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
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
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
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 16;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand
using         LayoutBTag  = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128;                                            // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::float_e4m3_t;                            // Element type for D matrix operand
using         ElementC    = void;                            // Element type for C matrix operand
using         ElementCSafe = cute::conditional_t<cute::is_void_v<ElementC>,half_t,ElementC>; // prevents void ref breakages
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementCSafe>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm100;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using MmaTileShape        = Shape<_256,_256,_128>;                          // MMA's tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,                      
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto or using targeted scheduling policy
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,                                                   // Indicates ProblemShape
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

//
// Data members
//

/// Initialization
StrideA stride_A;
LayoutA layout_A;
LayoutSFA layout_SFA;
StrideB stride_B;
LayoutB layout_B;
LayoutSFB layout_SFB;
StrideC stride_C;
LayoutC layout_C;
StrideD stride_D;
LayoutD layout_D;
uint64_t seed;

// The HostTensors are only used for allocating memory on host and device, and transferring data between host and device
// Use cute::Tensor and cute::Layout for iterating thru the matrix elements
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementCSafe, cutlass::layout::PackedVectorLayout> block_C;
// Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
// Reference Output Tensor
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_reference_D;
#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

template <typename T>
auto make_iterator(T* ptr) {
  using namespace cute;
  if constexpr (cute::is_subbyte_v<T>) {
    return subbyte_iterator<T>(ptr);
  }
  else {
    return ptr;
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  int swizzle = 0;
  int mask_ratio = 0;
  bool skip_verify;
  double scope_min;
  double scope_max;

  Options():
    help(false),
    m(8192), n(8192), k(8192),
    alpha(1.f), beta(0.f),
    iterations(10),
    swizzle(0),
    mask_ratio(0),
    skip_verify(false),
    scope_min(-5),
    scope_max(5)
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
    cmd.get_cmd_line_argument("skip_verify", skip_verify, false);
    cmd.get_cmd_line_argument("scope_min", scope_min, -5.0);
    cmd.get_cmd_line_argument("scope_max", scope_max, 5.0);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "72c_blackwell_mixed_mxfp8_bf16_gemm\n\n"
      << "  Blackwell Mxfp8 x Mxfp4 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n"
      << "  --swizzle=<int>             Cluster rasterization swizzle\n"
      << "  --mask_ratio=<int>           Percentage of elements to mask (set to zero) in A and B, default 0 (no mask)\n\n"
      << "  --scope_min=<float>         Minimum value for random initialization (default: -5)\n"
      << "  --scope_max=<float>         Maximum value for random initialization (default: 5)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --skip_verify=<bool>        If specified, skips verification step\n";

    out << "\n\nExamples:\n\n"
      << "$ " << "/examples/72_blackwell_narrow_precision_gemm/72c_blackwell_mixed_mxfp8_bf16_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

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
template <typename Element, typename Layout>
bool initialize_block(
  cutlass::TensorView<Element, Layout> view,
  uint64_t seed,
  double scope_min,
  double scope_max
) {

  constexpr int bits_input = cutlass::sizeof_bits<Element>::value;


  cutlass::reference::host::TensorFillRandomUniform(
    view, seed, scope_max, scope_min, 0);
  
  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  using namespace cute;
  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  layout_A = make_layout(make_shape(options.m, options.k, 1), stride_A);
  layout_B = make_layout(make_shape(options.n, options.k, 1), stride_B);
  layout_C = make_layout(make_shape(options.m, options.n, 1), stride_C);
  layout_D = make_layout(make_shape(options.m, options.n, 1), stride_D);
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));

  block_A.reset(cutlass::make_Coord(size(layout_A)));
  block_B.reset(cutlass::make_Coord(size(layout_B)));
  block_D.reset(cutlass::make_Coord(size(layout_D)));
  block_reference_D.reset(cutlass::make_Coord(size(layout_D)));
  block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

  initialize_block(block_A.host_view(), seed + 2021, options.scope_min, options.scope_max);
  initialize_block(block_B.host_view(), seed + 2022, options.scope_min, options.scope_max);
  initialize_block(block_SFA.host_view(), seed + 2024, options.scope_min, options.scope_max);
  initialize_block(block_SFB.host_view(), seed + 2025, options.scope_min, options.scope_max);

  if constexpr (not is_same_v<ElementC, void>) {
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    initialize_block(block_C.host_view(), seed + 2023, options.scope_min, options.scope_max);
  }

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

  auto mask_start_0 = std::chrono::high_resolution_clock::now();

  // Mask host_A and host_B if mask_ratio > 0
  if (options.mask_ratio > 0) {
    auto mask_start = std::chrono::high_resolution_clock::now();
    std::cout << "Masking " << options.mask_ratio << "% of the first tile in A and B matrices." << std::endl;
    float mask_ratio_f = options.mask_ratio / 100.0f;

    // Only mask the first tile
    size_t first_tile_size_A = tile_m * tile_k;
    size_t first_tile_size_B = tile_k * tile_n;

    size_t num_mask_A = static_cast<size_t>(first_tile_size_A * mask_ratio_f);
    size_t num_mask_B = static_cast<size_t>(first_tile_size_B * mask_ratio_f);

    std::vector<int> indices_A(first_tile_size_A);
    std::vector<int> indices_B(first_tile_size_B);

    std::iota(indices_A.begin(), indices_A.end(), 0);
    std::iota(indices_B.begin(), indices_B.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_A.begin(), indices_A.end(), g);
    std::shuffle(indices_B.begin(), indices_B.end(), g);

    // Mask elements in the first tile of A
    for (size_t i = 0; i < num_mask_A; ++i) {
      int idx = indices_A[i];
      block_A.at(cutlass::make_Coord(idx)) = typename Gemm::ElementA(0);
    }

    // Mask elements in the first tile of B
    for (size_t i = 0; i < num_mask_B; ++i) {
      int idx = indices_B[i];
      block_B.at(cutlass::make_Coord(idx)) = typename Gemm::ElementB(0);
    }
    std::cout << "Masked elements in A: " << num_mask_A << ", B: " << num_mask_B << std::endl;

    // After masking operations complete
    auto mask_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> mask_duration = mask_end - mask_start;
    std::cout << "Time taken for masking: " << mask_duration.count() << " ms" << std::endl;
  }

  for (int m = 0; m < options.m; ++m) {
    for (int k = 0; k < options.k; ++k) {
      if (m < tile_m && k < tile_k) continue;
      block_A.at(cutlass::make_Coord(m * options.k + k)) = block_A.at(cutlass::make_Coord((m % tile_m) * options.k + (k % tile_k)));
    }
  }

  for (int k = 0; k < options.k; ++k) {
    for (int n = 0; n < options.n; ++n) {
      if (k < tile_k && n < tile_n) continue;
      block_B.at(cutlass::make_Coord(k * options.n + n)) = block_B.at(cutlass::make_Coord((k % tile_k) * options.n + (n % tile_n)));
    }
  }

  auto mask_end_0 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> mask_duration_0 = mask_end_0 - mask_start_0;
  std::cout << "Time taken for manipulating: " << mask_duration_0.count() << " ms" << std::endl << std::flush;

  block_A.sync_device();
  block_B.sync_device();
  block_SFA.sync_device();
  block_SFB.sync_device();

  if constexpr (not is_same_v<ElementC, void>) {
    block_C.sync_device();
  }
}

// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  auto ptr_C = [&]() {
    if constexpr (not is_same_v<ElementC, void>) {
      return block_C.device_data();
    }
    else {
      return static_cast<ElementD const*>(nullptr); // C is not used in this case
    }
  }();

  typename Gemm::Arguments arguments {
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k, 1},
    { // Mainloop arguments
      block_A.device_data(), stride_A,
      block_B.device_data(), stride_B,
      block_SFA.device_data(), layout_SFA,
      block_SFB.device_data(), layout_SFB
    },
    { // Epilogue arguments
      {options.alpha, options.beta},
      ptr_C, stride_C,
      block_D.device_data(), stride_D
    }
  };

  arguments.scheduler.max_swizzle_size = options.swizzle;
  return arguments;
}

bool verify(const Options &options) {
  using namespace cute;
  // Create the arguments for host reference implementation
  Tensor tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
  Tensor tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
  Tensor tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
  Tensor tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);

  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAccumulator,                 // ElementAccumulator
      decltype(tensor_A),                 // TensorA
      decltype(tensor_SFA),               // TensorSfA
      decltype(tensor_B),                 // TensorB
      decltype(tensor_SFB)                // TensorSfB
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

  auto tensor_C = [&]() {
    if constexpr (not is_same_v<ElementC, void>) {
      return cute::make_tensor(make_iterator(block_C.host_data()), layout_C);
    }
    else {
      return cute::make_tensor(make_iterator(static_cast<ElementD const*>(nullptr)), layout_C);
    }
  }();

  auto tensor_D = cute::make_tensor(make_iterator(block_reference_D.host_data()), layout_D);
 
  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAccumulator,                   // ElementScalar
      ElementAccumulator,                   // ElementAccumulator
      ElementAccumulator,                   // ElementCompute
      decltype(tensor_C),                   // TensorC
      decltype(tensor_D)                    // TensorD
    > epilogue_params{options.alpha, options.beta, tensor_C, tensor_D};

  cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

  // Comparison
  block_D.sync_host();
  bool passed = cutlass::reference::host::TensorEquals(block_reference_D.host_view(), block_D.host_view());
  passed &= (cutlass::reference::host::TensorNorm(block_reference_D.host_view()) > 0);
  passed &= (cutlass::reference::host::TensorNorm(block_D.host_view()) > 0);

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  auto init_start = std::chrono::high_resolution_clock::now();
  initialize(options);
  auto init_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> init_duration = init_end - init_start;
  std::cout << "Initialization completed in " << init_duration.count() << " ms" << std::endl;

  auto gemm_init_start = std::chrono::high_resolution_clock::now();

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

  auto gemm_init_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> gemm_init_duration = gemm_init_end - gemm_init_start;
  std::cout << "Gemm initialization completed in " << gemm_init_duration.count() << " ms" << std::endl;

  auto gemm_warmup_start = std::chrono::high_resolution_clock::now();

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run());

  cudaDeviceSynchronize();

  auto gemm_warmup_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> gemm_warmup_duration = gemm_warmup_end - gemm_warmup_start;
  std::cout << "Gemm warmup completed in " << gemm_warmup_duration.count() << " ms" << std::endl;

  Result result;

  if (!options.skip_verify) {  // 根据skip_verify条件跳过验证
    auto verify_start = std::chrono::high_resolution_clock::now();
    // Check if output from CUTLASS kernel and reference kernel are equal or not
    result.passed = verify(options);
    auto verify_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> verify_duration = verify_end - verify_start;
    std::cout << "Verification completed in " << verify_duration.count() << " ms" << std::endl;

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
      exit(-1);
    }
  } else {
    std::cout << "  Disposition: " << "Skipped" << std::endl;
  }

  auto get_timestamp = []() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream timestamp;
    timestamp << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << now_ms.count();
    return timestamp.str();
  };

  std::cout << "  [" << get_timestamp() << "] "
            << "Start profiling CUTLASS kernel for " << options.iterations << " iterations." << std::endl;

  // Run profiling loop
  if (options.iterations > 0)
  {
    auto profiling_start = std::chrono::high_resolution_clock::now();
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
    auto profiling_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> profiling_duration = profiling_end - profiling_start;
    std::cout << "Profiling completed in " << profiling_duration.count() << " ms." << std::endl;

    std::cout << "  [" << get_timestamp() << "] " << "Profiling completed. Results:" << std::endl;
    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  MaskRatio: " << options.mask_ratio << "%" << std::endl;
    std::cout << "  ScopeRange: [" << options.scope_min << ", " << options.scope_max << "]" << std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;
    std::cout << "  TFLOPS: " << result.gflops / 1000.0 << std::endl;
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.8 or higher Toolkit to run this example
  // and must have compute capability at least 100.
  if (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)) {
    std::cerr << "This example requires CUDA 12.8 or newer." << std::endl;
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  
  if (!(props.major == 10 && props.minor == 0)) {
    std::cerr << "This example requires a GPU of NVIDIA's Blackwell architecture (compute capability 100)." << std::endl;
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
