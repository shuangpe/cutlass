# SM90 Blockwise Scaling 和 Groupwise Scaling 实现分析

## 背景与意义

在NVIDIA Hopper架构(SM90)上进行FP8(8位浮点数)GEMM计算时，由于精度有限，通常需要缩放因子(scaling factor)来扩展其动态范围。缩放因子可以应用在不同的粒度级别上，以平衡精度和性能。本文档分析两种主要的缩放策略实现：

- **Blockwise Scaling**：对矩阵的大块区域应用相同的缩放因子，内存占用小，但精度可能较低
- **Groupwise Scaling**：对矩阵的小组(甚至单行)应用不同的缩放因子，精度更高，但内存占用和计算开销增大

## 基础概念

Blockwise scaling和groupwise scaling是针对FP8 GEMM操作提高计算精度的技术，通过为矩阵块或组应用不同的缩放因子来扩展有效的动态范围。

**关键术语**：
- **TMA** (Tensor Memory Accelerator)：Hopper架构引入的用于高效传输大块数据的硬件单元
- **CP.ASYNC**：用于异步数据加载的指令，适合加载较小数据块
- **ScaleGranularity**：缩放因子应用的粒度，例如每128行共用一个因子或每行一个因子

## ScaleConfig配置

### Blockwise配置实现

在example文件[67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu](../examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling.cu#L127)中，`ScaleConfig`定义如下：

```c++
using ScaleConfig = decltype(cutlass::detail::sm90_trivial_blockwise_scale_config(TileShape{}));

using LayoutSFA             = decltype(ScaleConfig::deduce_layoutSFA());                     // Layout type for SFA matrix operand
using LayoutSFB             = decltype(ScaleConfig::deduce_layoutSFB());                     // Layout type for SFB matrix operand
```

`sm90_trivial_blockwise_scale_config`函数定义在[include/cutlass/detail/blockwise_scale_layout.hpp](../include/cutlass/detail/blockwise_scale_layout.hpp#L282)中：

```c++
// Sm90只支持SFA和SFB的MN major布局
template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK>
using Sm90BlockwiseScaleConfig = Sm100BlockwiseScaleConfig<SFVecSizeM, SFVecSizeN, SFVecSizeK>;

template<class MmaTileShape_MNK>
constexpr auto sm90_trivial_blockwise_scale_config(MmaTileShape_MNK) {
  return Sm90BlockwiseScaleConfig<size<0>(MmaTileShape_MNK{}), size<1>(MmaTileShape_MNK{}), size<2>(MmaTileShape_MNK{})>{};
}
```

这个函数接收TileShape作为参数，然后返回一个`Sm90BlockwiseScaleConfig`实例，其模板参数是TileShape的M、N、K维度大小。在"trivial"配置中，每个缩放因子覆盖整个tile的相应维度，这意味着：
- 对于M维度，缩放粒度等于整个tile的M大小
- 对于N维度，缩放粒度等于整个tile的N大小
- 对于K维度，缩放粒度等于整个tile的K大小

例如，对于一个典型的TileShape = Shape<_128,_128,_128>，trivial blockwise配置会生成Sm90BlockwiseScaleConfig<_128,_128,_128>。

### Groupwise配置实现

在[67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu](../examples/67_hopper_fp8_warp_specialized_gemm_with_blockwise_scaling/67_hopper_fp8_warp_specialized_gemm_with_groupwise_scaling.cu#L128-L132)中，我们看到了更细粒度的groupwise配置：

```c++
constexpr int ScaleGranularityM = 1;  // 每个M行有独立的缩放因子 (更细粒度)
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;
```

与blockwise配置不同，groupwise配置在特定维度上使用了较小的缩放粒度，从而提供更灵活的缩放控制。在这个例子中，M维度上的缩放粒度为1，意味着每一行都有独立的缩放因子。

### Blockwise与Groupwise的区别

Blockwise scaling和Groupwise scaling的主要区别在于缩放因子的粒度设置：

- **Blockwise scaling**：通常使用较粗的粒度，例如一个完整的tile共用一个缩放因子。在典型配置中，缩放因子的粒度等于tile的完整维度大小。

- **Groupwise scaling**：使用更细的粒度，尤其是在某些特定维度上。通过减小某些维度的缩放粒度（如ScaleGranularityM=1），可以为每个更小的子块提供独立的缩放因子。

重要的是，除了ScaleConfig配置的不同外，Blockwise和Groupwise scaling使用完全相同的底层代码实现。它们共享相同的内存布局策略、TMA/CP.ASYNC选择逻辑以及缩放因子应用机制，只是在缩放粒度的定义上有所不同。这种设计使得代码可以灵活地适应不同的精度需求，而无需更改核心实现。

### LayoutSFA和LayoutSFB的结构

`LayoutSFA`和`LayoutSFB`通过`deduce_layoutSFA()`和`deduce_layoutSFB()`方法从`ScaleConfig`中推导出来。这些是描述缩放因子在内存中组织方式的布局类型。

```cpp
template<int SFVecSizeM, int SFVecSizeN, int SFVecSizeK, UMMA::Major majorSFA = UMMA::Major::MN, UMMA::Major majorSFB = UMMA::Major::MN>
struct Sm100BlockwiseScaleConfig {
  using ShapeSFA = Shape<Shape<Int<SFVecSizeM>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;
  using ShapeSFB = Shape<Shape<Int<SFVecSizeN>, int32_t>, Shape<Int<SFVecSizeK>, int32_t>, int32_t>;

  // stride中的`_0`表示同一维度内重用同一个值
  using StrideSFA = conditional_t<majorSFA == UMMA::Major::MN,
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>,
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using StrideSFB = conditional_t<majorSFB == UMMA::Major::MN,
      Stride<Stride<_0,_1>,Stride<_0,int32_t>, int32_t>,
      Stride<Stride<_0,int32_t>,Stride<_0,_1>, int32_t>>;

  using LayoutSFA = Layout<ShapeSFA, StrideSFA>;
  using LayoutSFB = Layout<ShapeSFB, StrideSFB>;
}
```

#### Blockwise配置（ScaleGranularityM=128, ScaleGranularityN=128, ScaleGranularityK=128）:

```cpp
BlockwiseScaleConfig = Sm100BlockwiseScaleConfig<size<0>(TileShape{}), size<1>(TileShape{}), size<2>(TileShape{})>{};
```

#### Groupwise配置（ScaleGranularityM=1, ScaleGranularityN=128, ScaleGranularityK=128）:

```cpp
GroupwiseScaleConfig = Sm100BlockwiseScaleConfig<1, size<1>(TileShape{}), size<2>(TileShape{})>{};
```

在M维度上，每个元素都有自己的缩放因子(粒度为1)，所以shape的第一部分是`Shape<_1, int32_t>`，表示每个元素一个缩放因子。

## Warp Specialization设计

SM90 Blockwise/Groupwise Scaling实现采用了Warp Specialized设计模式，将线程块内的warp分为不同角色执行不同任务。从`sm90_gemm_tma_warpspecialized_cooperative.hpp`的代码骨架中可以看到这种分工模式：

```cpp
enum class WarpGroupRole {
  Producer = 0,
  Consumer0 = 1,
  Consumer1 = 2
};

enum class ProducerWarpRole {
  Mainloop = 0,
  Warp1 = 1,
  Epilogue = 2,
  MainloopAux = 3
};

int warp_idx = canonical_warp_idx_sync();
int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);

if (warp_group_role == WarpGroupRole::Producer) {
  if (producer_warp_role == ProducerWarpRole::Warp1) {
    scheduler.advance_to_next_work(...);
  }
  else if (producer_warp_role == ProducerWarpRole::Mainloop) {
    collective_mainloop.load(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
      blk_coord, k_tile_iter, work_k_tile_count, lane_idx, block_rank_in_cluster, shared_storage.tensors.mainloop);
  }
  else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
    if constexpr (IsMainloopAuxiliaryLoadNeeded) {
      collective_mainloop.load_auxiliary(params.mainloop, mainloop_pipeline, mainloop_pipe_producer_state, load_inputs,
        blk_coord, k_tile_iter, work_k_tile_count, lane_idx, block_rank_in_cluster, shared_storage.tensors.mainloop);
    }
  }
  else if (producer_warp_role == ProducerWarpRole::Epilogue && is_epi_load_needed) {
    collective_epilogue.load(...);
  }
}
else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
  collective_mainloop.mma(...);
}
```

## 内存布局

### Global Memory 布局

在初始化函数中，使用`tile_atom_to_shape_SFA()`和`tile_atom_to_shape_SFB()`函数来设置全局内存布局：

```c++
layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(options.m, options.n, options.k, options.l));
layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(options.m, options.n, options.k, options.l));
```

`tile_atom_to_shape_SFA`函数的实现如下（以SFA为例，SFB实现类似）：

```c++
template <class ProblemShape>
CUTE_HOST_DEVICE
static constexpr auto
tile_atom_to_shape_SFA(ProblemShape const& problem_shape) {
  auto problem_shape_MNKL = append<4>(problem_shape, 1);

  auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
    auto [M, N, K, L] = problem_shape_MNKL;
    if constexpr (majorSFA == UMMA::Major::MN) {
      return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, cute::ceil_div(M, SFVecSizeM)));
    }
    else {
      return make_stride(make_stride(_0{}, cute::ceil_div(K, SFVecSizeK)), make_stride(_0{}, _1{}));
    }
  }();

  auto [M, N, K, L] = problem_shape_MNKL;
  auto mk_layout = make_layout(
    make_shape(make_shape(Int<SFVecSizeM>{}, cute::ceil_div(M, SFVecSizeM)),
              make_shape(Int<SFVecSizeK>{}, cute::ceil_div(K, SFVecSizeK))),
    strides
  );

  return make_layout(append(shape(mk_layout), L), append(stride(mk_layout), size(filter_zeros(mk_layout))));
}
```

完整实现可参考[blockwise_scale_layout.hpp](../include/cutlass/detail/blockwise_scale_layout.hpp)。

对于TileShape = Shape<_128,_128,_128>, ProblemShape = (256, 512, 1024)的情况：

#### Blockwise配置（ScaleGranularityM=128, ScaleGranularityN=128, ScaleGranularityK=128）:
- layout_SFA: ((_128,2), (_128,8), 1) :((_0,_1), (_0,2), 2*8)
- layout_SFB: ((_128,4), (_128,8), 1) :((_0,_1), (_0,4), 4*8)

这意味着：
- 整个M=256被划分为2个区域，每个区域128行共享一个缩放因子
- 整个N=512被划分为4个区域，每个区域128列共享一个缩放因子
- 整个K=1024被划分为8个区域，每个区域128深度共享一个缩放因子

#### Groupwise配置（ScaleGranularityM=1, ScaleGranularityN=128, ScaleGranularityK=128）:
- layout_SFA: ((_1,256), (_128,8), 1) :((_0,_1), (_0,256), 256*8)
- layout_SFB: ((_128,4), (_128,8), 1) :((_0,_1), (_0,  4),   4*8)

### Shared Memory 布局

在定义共享内存布局时，首先创建基础的原子布局：

```c++
using SmemLayoutAtomSFA = decltype(ScaleConfig::smem_atom_layoutSFA(TileShape{}));
using SmemLayoutAtomSFB = decltype(ScaleConfig::smem_atom_layoutSFB(TileShape{}));
```

`smem_atom_layoutSFA`函数的实现如下（以SFA为例）：

```c++
template<typename CtaShape_MNK>
CUTE_HOST_DEVICE
static constexpr auto
smem_atom_layoutSFA(CtaShape_MNK cta_shape_mnk) {
  static_assert(cute::is_static_v<CtaShape_MNK>, "Expect static CTA shape");
  auto strides = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
    auto [M, N, K] = cta_shape_mnk;
    if constexpr (majorSFA == UMMA::Major::MN) {
      return make_stride(make_stride(_0{}, _1{}), make_stride(_0{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}));
    }
    else {
      return make_stride(make_stride(_0{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{}), make_stride(_0{}, _1{}));
    }
  }();

  auto [M, N, K] = cta_shape_mnk;
  return make_layout(
    make_shape(make_shape(Int<SFVecSizeM>{}, Int<cute::ceil_div(size<0>(CtaShape_MNK{}), SFVecSizeM)>{}),
               make_shape(Int<SFVecSizeK>{}, Int<cute::ceil_div(size<2>(CtaShape_MNK{}), SFVecSizeK)>{})),
    strides
  );
}
```

完整实现可参考[blockwise_scale_layout.hpp](../include/cutlass/detail/blockwise_scale_layout.hpp)。

对于TileShape = Shape<_128,_128,_128>，得到的原子布局为：

#### Blockwise配置:
```
SmemLayoutAtomSFA{}: ((_128,_1),(_128,_1)):((_0,_1),(_0,_1))
SmemLayoutAtomSFB{}: ((_128,_1),(_128,_1)):((_0,_1),(_0,_1))
```

#### Groupwise配置:
```
SmemLayoutAtomSFA{}: ((_1,_128),(_128,_1)):((_0,_1),(_0,_128))
SmemLayoutAtomSFB{}: ((_128,_1),(_128,_1)):((_0,_1),(_0,_1))
```

然后，通过添加pipeline stages维度扩展为最终的共享内存布局：

```c++
using SmemLayoutSFA = decltype(make_layout(
  append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
  append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
));
```

对于DispatchPolicy::Stages = 4，得到的最终布局为：

#### Blockwise配置:
```
SmemLayoutSFA{}: ((_128,_1),(_128,_1),_4):((_0,_1),(_0,_1),_1)
SmemLayoutSFB{}: ((_128,_1),(_128,_1),_4):((_0,_1),(_0,_1),_1)
```

#### Groupwise配置:
```
SmemLayoutSFA{}: ((_1,_128),(_128,_1),_4):((_0,_1),(_0,_128),_128)
SmemLayoutSFB{}: ((_128,_1),(_128,_1),_4):((_0,_1),(_0,_1),_1)
```

## 数据准备和任务划分

SM90实现使用精心设计的数据准备和任务划分机制来处理缩放因子。这一流程包含以下几个关键步骤：

### to_underlying_arguments：参数准备

[to_underlying_arguments](../include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp#L230-L290)函数负责将用户输入的参数转换为内部实现所需的格式，特别是为缩放因子创建高效的TMA描述符：

```c++
template <class ProblemShape>
static constexpr Params
to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {

  // tensor_sfa: ((_1,256),(_1,8),1):((_0,_1),(_0,256),2048)
  // tensor_sfb: ((_1,  4),(_1,8),1):((_0,_1),(_0,  4),  32)
  Tensor tensor_sfa = make_tensor(ptr_SFA, filter_zeros(args.layout_SFA));
  Tensor tensor_sfb = make_tensor(ptr_SFB, filter_zeros(args.layout_SFB));

  // tma_load_sfa: TiledCopy
  //   Tiler_MN:       (_128,_1)
  //   TiledLayout_TV: (_1,((_128,_1))):(_0,((_1,_0)))
  // Copy_Atom
  //   ThrID:        _1:_0
  //   ValLayoutSrc: (_1,_128):(_0,_1)
  //   ValLayoutDst: (_1,_128):(_0,_1)
  //   ValLayoutRef: (_1,_128):(_0,_1)
  //   ValueType:    32b
  if constexpr (IsTmaLoadSFA) {
    tma_load_sfa = make_tma_copy(
        GmemTiledCopyScaleTMA{},
        tensor_sfa,
        filter_zeros(SmemLayoutSFA{})(_,_,cute::Int<0>{}),
        Shape<Int<ScaleMsPerTile>, Int<1>>{},
        _1{});
  }

  //tma_load_sfb: (null)
  typename Params::TMA_SFB tma_load_sfb{};
  if constexpr (IsTmaLoadSFB) {
    tma_load_sfb = make_tma_copy(...);
  }
}
```

## 数据拷贝过程

数据从global memory到shared memory的拷贝分为两种方式：
1. 使用TMA (Tensor Memory Accelerator)：Hopper架构的新单元，可高效传输大块数据
2. 使用CP.ASYNC指令：适合传输较小数据块的异步指令
    ```c++
    // Block scaling gmem-to-smem copy atom
    //  we can have partial tiles in M or N, so don't vectorize those loads
    using CopyAtomSFA = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
    using CopyAtomSFB = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<ElementBlockScale>, ElementBlockScale>;
    ```

在[sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp](../include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp#L124-L134)中，根据scale tile的大小决定是否使用TMA：

```c++
static constexpr int ScaleTmaThreshold = 32;
static constexpr bool IsTmaLoadSFA = ScaleMsPerTile >= ScaleTmaThreshold && ScaleNsPerTile < ScaleTmaThreshold;
static constexpr bool IsTmaLoadSFB = ScaleNsPerTile >= ScaleTmaThreshold && ScaleMsPerTile < ScaleTmaThreshold;
// Two threads per CTA are producers (1 for operand tile `tma`, and 32 for scales `cp.async`)
static constexpr int NumProducerThreadEvents = ((IsTmaLoadSFA && IsTmaLoadSFB)? 1 : 33);
```

当scale维度足够大时（≥32），使用TMA进行高效加载；否则使用CP.ASYNC。

### load：加载主矩阵和缩放因子

`load`函数主要负责使用TMA加载主矩阵数据和缩放因子(当符合TMA加载条件时)：

```c++
CUTLASS_DEVICE void
load(...) {
  // mSFA_mkl: ArithTuple(_0,_0,_0) o ((_1,256),(_1,8),1):((_0,_1@0),(_0,_1@1),_1@2)
  //     gSFA: ArithTuple( 0,_0, 0) o (_128,_1,8):(_1@0,_0,_1@1)
  Tensor gSFA = local_tile(mSFA_mkl, make_tile(Int<ScaleMsPerTile>{}, Int<1>{}), make_coord(m_coord,_,l_coord));

  // mSFB_nkl: gmem_ptr[32b](0x7f4ad06c2000) o ((_128,4),(_128,8),1):((_0,_1),(_0,4),32)
  //     gSFB: gmem_ptr[32b](0x7f4ad06c2000) o (_1,_1,(_128,8)):(_0,_0,(_0,4))
  Tensor gSFB = local_tile(mSFB_nkl, make_tile(Int<ScaleNsPerTile>{}, Int<1>{}), make_coord(n_coord,_,l_coord));

  // tAgA_SFA:            ArithTuple(0,_0,0) o ((_128,_1),_1,_1, 8):((_1@0,_0),_0,_0,_1@1)
  // tAsA_SFA: smem_ptr[32b](0x7f4b00030800) o ((_128,_1),_1,_1,_3):((_1,  _0),_0,_0,_128)
  auto [tAgA_SFA, tAsA_SFA] = [&]() {
    if constexpr (IsTmaLoadSFA) {
      auto block_tma_sfa = mainloop_params.tma_load_sfa.get_slice(cluster_local_block_id.y);
      Tensor tAgA_SFA_ = block_tma_sfa.partition_S(gSFA);
      Tensor tAsA_SFA_ = block_tma_sfa.partition_D(sSFA);
      return cute::make_tuple(tAgA_SFA_, tAsA_SFA_);
    }
    else {
      return cute::make_tuple(0, 0);
    }
  }();

  // tBgB_SFB: 0
  // tBsB_SFB: 0
  auto [tBgB_SFB, tBsB_SFB] = [&]() {
    if constexpr (IsTmaLoadSFB) {
      auto block_tma_sfb = mainloop_params.tma_load_sfb.get_slice(cluster_local_block_id.x);
      Tensor tBgB_SFB_ = block_tma_sfb.partition_S(gSFB);
      Tensor tBsB_SFB_ = block_tma_sfb.partition_D(sSFB);
      return cute::make_tuple(tBgB_SFB_, tBsB_SFB_);
    }
    else {
      return cute::make_tuple(0, 0);
    }
  }();

  CUTLASS_PRAGMA_NO_UNROLL
  for ( ; k_tile_count > 0; --k_tile_count) {
    if (lane_predicate && IsTmaLoadSFA) {
      copy(mainloop_params.tma_load_sfa.with(*tma_barrier, mcast_mask_sf), tAgA_SFA(_,_,_,*k_tile_iter), tAsA_SFA(_,_,_,write_stage));
    }
  }
}
```

### load_auxiliary：加载辅助缩放因子

`load_auxiliary`函数负责使用CP.ASYNC加载缩放因子(当不适合TMA加载条件时)：

```c++
// load_auxiliary函数处理辅助缩放因子加载
CUTLASS_DEVICE void
load_auxiliary(...) {
  // scale_copy_b: TiledCopy
  //   Tiler_MN:       (_32)
  //   TiledLayout_TV: (_32,_1):(_1,_0)
  // Copy_Atom
  //   ThrID:        _1:_0
  //   ValLayoutSrc: (_1,_1):(_0,_1)
  //   ValLayoutDst: (_1,_1):(_0,_1)
  //   ValLayoutRef: (_1,_1):(_0,_1)
  //   ValueType:    32b
  TiledCopy scale_copy_b = make_tiled_copy(CopyAtomSFB{},
    Layout<Shape<_32>>{}, Layout<Shape<_1>>{});
  ThrCopy thr_scale_copy_b = scale_copy_b.get_slice(thread_idx);

  // mSFB_nkl: gmem_ptr[32b](0x7f999c6c2000) o ((_128,4),(_128,8),1):((_0,_1),(_0,4),32)
  // gSFB_nkl: gmem_ptr[32b](0x7f999c6c2000) o (_128,_128,4,8,1):(_0,_0,_1,4,32)
  // gSFB_k: gmem_ptr[32b](0x7f999c6c2000) o (_128,_128,8):(_0,_0,4)
  // tSFBgSFB_k: gmem_ptr[32b](0x7f999c6c2000) o ((_1,_1),_4,_128,8):((_0,_0),_0,_0,4)
  // filter_zeros(tSFBgSFB_k(_,_,_,0)): gmem_ptr[32b](0x7f999c6c2000) o ((_1,_1),_1,_1):((_0,_0),_0,_0)
  Tensor gSFB_nkl = local_tile(mSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
  Tensor gSFB_k = gSFB_nkl(_,_,n_coord,_,l_coord);
  Tensor tSFBgSFB_k = thr_scale_copy_b.partition_S(gSFB_k);

  // iSFB_nkl: ArithTuple((_0,_0),(_0,_0),_0) o ((_128,4),(_128,8),1):((_1@0@0,_1@1@0),(_1@0@1,_1@1@1),_1@2)
  // cSFB_nkl: ArithTuple((_0,_0),(_0,_0),_0) o (_128,_128,4,8,1):(_1@0@0,_1@0@1,_1@1@0,_1@1@1,_1@2)
  // cSFB_k: ArithTuple((_0,0),(_0,_0),0) o (_128,_128,8):(_1@0@0,_1@0@1,_1@1@1)
  // tSFBcSFB_k: ArithTuple((0,0),(_0,_0),0) o ((_1,_1),_4,_128,8):((_0,_0),_0,_0,4)
  // filter_zeros(tSFBcSFB_k(_,_,_,0)): ArithTuple((0,0),(_0,0),0) o ((_1,_1),_4,_128):((_0,_0),_0,_0)
  Tensor iSFB_nkl = make_identity_tensor(shape(mainloop_params.layout_SFB));    // (n,k,l)
  Tensor cSFB_nkl = local_tile(iSFB_nkl, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});     // (BLK_N,BLK_K,n,k,l)
  Tensor cSFB_k = cSFB_nkl(_,_,n_coord,_,l_coord);
  Tensor tSFBcSFB_k = thr_scale_copy_b.partition_S(cSFB_k);

  // sSFB:     smem_ptr[32b](0x7f9a00030e00) o ((_128,_1),(_128,_1),_3):((_0,_1),(_0,_1),_1)
  // tSFBsSFB: smem_ptr[32b](0x7f9a00030e00) o ((_1,_1),_4,(_128,_1),_3):((_0,_0),_0,(_0,_1),_1)
  // tSFBpSFB:  subptr[1b](0x7f99adfffcac.0) o ((_1,_1),_1,(_1,_1)):((_0,_0),_0,(_0,_0))
  Tensor sSFB = make_tensor(cute::make_smem_ptr(shared_tensors.smem_SFB.data()), SmemLayoutSFB{}); // (ScaleNsPerTile,k)
  Tensor tSFBsSFB = thr_scale_copy_b.partition_D(sSFB);
  Tensor tSFBpSFB = make_tensor<bool>(shape(filter_zeros(tSFBsSFB(_,_,_,_0{}))));                 // (CPY,CPY_N,CPY_K)

  // SFB_shape: ((_128,4),(_128,8),1)
  auto SFB_shape = shape(mainloop_params.layout_SFB);

  // Mainloop
  CUTLASS_PRAGMA_NO_UNROLL
  for ( ; k_tile_count > 0; --k_tile_count) {
    // ScaleNsPerTile: 1
    // tSFBcSFB: ArithTuple((0,0),(_0,0),0) o ((_1,_1),_4,_128):((_0,_0),_32@0@0,_1@0@1)
    // tSFBcSFB_compact: ArithTuple((0,0),(_0,0),0) o ((_1,_1),_4,_128):((_0,_0),_32@0@0,_1@0@1)
    // size(tSFBpSFB): _1
    // get<0>(tSFBcSFB_compact(0)): (0,0)
    bool load_sfb = thread_idx < ScaleNsPerTile;
    Tensor tSFBcSFB = tSFBcSFB_k(_,_,_,*k_tile_iter);
    Tensor tSFBcSFB_compact = filter_zeros(tSFBcSFB);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tSFBpSFB); ++i) {
      tSFBpSFB(i) = load_sfb && elem_less(get<0>(tSFBcSFB_compact(i)), get<0>(SFB_shape));
    }

    // filter_zeros(tSFBgSFB_k(_,_,_,0)): gmem_ptr[32b](0x7efc106c2000) o ((_1,_1),_1,_1):((_0,_0),_0,_0)
    // filter_zeros(tSFBsSFB(_,_,_,0)): smem_ptr[32b](0x7efd00030e00) o ((_1,_1),_1,(_1,_1)):((_0,_0),_0,(_0,_1))
    if constexpr (!IsTmaLoadSFB) {
      copy_if(scale_copy_b, tSFBpSFB, filter_zeros(tSFBgSFB_k(_,_,_,*k_tile_iter)), filter_zeros(tSFBsSFB(_,_,_,write_stage)));
    }
  }
}
```

### mma：执行计算和缩放

在MMA函数中，应用缩放因子到计算结果：

```c++
CUTLASS_DEVICE void
mma(...) {
  // sSFA: smem_ptr[32b](0x7f3b00030800) o ((_1,_128),_128,((_128,_1),_3)):((_0,_1),_0,((_0,_128),_128))
  // tCsSFA: smem_ptr[32b](0x7f3b00030900) o ((_2,_2,_16),_1,_1,((_128,_1),_3)):((_0,_8,_0),_0,_0,((_0,_128),_128))
  Tensor tCsSFA = tiled_mma.get_slice(thread_idx).partition_C(sSFA);  // (MMA,MMA_M,MMA_N,(MMA_K,PIPE))

  // sSFB: smem_ptr[32b](0x7f3b00030e00) o (_128,(_128,_1),((_128,_1),_3)):(_0,(_0,_1),((_0,_1),_1))
  // tCsSFB: smem_ptr[32b](0x7f3b00030e00) o ((_2,_2,_16),_1,_1,((_128,_1),_3)):((_0,_0,_0),_0,_0,((_0,_1),_1))
  Tensor tCsSFB = tiled_mma.get_slice(thread_idx).partition_C(sSFB);

  // tCrSFA: ptr[32b](0x7f3aa9fffba0) o ((_2,_2,_16),_1,_1):((_0,_1,_0),_0,_0)
  // tCrSFB: ptr[32b](0x7f3aa9fffbb0) o ((_2,_2,_16),_1,_1):((_0,_0,_0),_0,_0)
  Tensor tCrSFA = make_tensor_like<ElementBlockScale>(tCsSFA(_, _, _, _0{}));  // (MMA,MMA_M,MMA_N)
  Tensor tCrSFB = make_tensor_like<ElementBlockScale>(tCsSFB(_, _, _, _0{}));  // (MMA,MMA_M,MMA_N)

  // accum: ptr[32b](0x7f3aa9fffbc0) o ((_2,_2,_16),_1,_1):((_1,_2,_4),_0,_0)
  GmmaFP8Accumulation(accumulation, ScalePromotionInterval, size<2>(tCrA));

  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accumulation());
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }

  copy(tCsSFA(_,_,_,make_coord(_0{}, read_stage)), tCrSFA);
  copy(tCsSFB(_,_,_,make_coord(_0{}, read_stage)), tCrSFB);

  if constexpr (ScaleMsPerTile == 1 && ScaleNsPerTile == 1) {
    tCrSFA(_0{}) = tCrSFA(_0{}) * tCrSFB(_0{});
    ElementBlockScale scale_ab = tCrSFA(_0{});
    scale_if_needed(accumulation, scale_ab);
  } else if constexpr (ScaleMsPerTile > 1 && ScaleNsPerTile == 1) {
    ElementBlockScale scale_b = tCrSFB(_0{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(filter_zeros(tCrSFA)); i++) {
      filter_zeros(tCrSFA)(i) = filter_zeros(tCrSFA)(i) * scale_b;
    }
    scale_if_needed(accumulation, tCrSFA);
  } else if constexpr (ScaleMsPerTile > 1 && ScaleNsPerTile > 1) {
    scale_if_needed(accumulation, tCrSFA, tCrSFB);
  }
}
```

## 选择合适的缩放策略

在实际应用中，如何选择Blockwise或Groupwise缩放策略取决于以下因素：

1. **精度需求**：
   - 如果需要更高精度，特别是当矩阵数据在不同行之间变化很大时，选择Groupwise缩放
   - 如果精度要求适中，并且数据分布相对均匀，Blockwise缩放可能足够

2. **性能考量**：
   - Blockwise缩放内存占用更小，数据传输和处理开销更低
   - Groupwise缩放(特别是M维度为1)需要更多内存和处理时间

3. **模型特性**：
   - 对于Transformer等模型，不同序列位置可能有显著不同的数值特征，适合用Groupwise缩放
   - 对于CNN等模型，相邻区域特征相似，可能Blockwise缩放已足够

## 内存优化和性能考虑

1. 对于缩放因子的加载，根据大小选择合适的拷贝方式（TMA或CP.ASYNC）
2. 使用预定义的阈值`ScaleTmaThreshold`进行判断
3. 共享内存布局考虑了bank冲突问题：
```c++
CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
CUTE_ALIGNAS(128) cute::array<ElementBlockScale, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
```

## 结论

SM90 (Hopper架构) 中的Blockwise和Groupwise scaling实现通过以下方式优化FP8 GEMM运算：

1. **灵活的缩放粒度**：提供从粗粒度到细粒度的缩放选项，适应不同精度需求
2. **智能的内存访问优化**：根据缩放因子数量自动选择最优的内存加载方式(TMA或CP.ASYNC)
3. **高效的内存布局**：通过精心设计的内存组织减少冲突，提高访问效率
4. **计算流水线优化**：与GEMM计算紧密集成，最小化性能开销

通过这些技术，开发者可以根据应用需求灵活选择缩放策略，在精度和性能之间取得平衡，充分发挥Hopper架构的FP8计算能力。