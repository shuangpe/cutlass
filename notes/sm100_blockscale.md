# SM100 Block Scaling in CUTLASS

## 背景与意义

NVIDIA Blackwell架构引入了全新的Block Scaled Tensor Core指令，提供了对低精度数据类型（如4位、6位和8位浮点数）的支持，同时通过缩放因子机制保证计算精度。这些指令不仅比Hopper架构高2-4倍的理论性能，还为AI训练和推理提供了更高效的低精度计算能力。本文档分析CUTLASS库中对于Blackwell SM100 block scaling特性的实现，特别是针对新的`tcgen05.mma.block_scale`指令集。

## SM100 Block Scaling基础概念

Blackwell SM100的Block Scaling是一种为低精度数据类型提供动态范围扩展的技术，通过为K维度上的数据块应用缩放因子来提高计算精度。与Hopper的FP8实现相比，Blackwell的Block Scaling特性有以下关键特点：

1. **专用硬件指令**: `tcgen05.mma.block_scale`指令系列专门设计用于支持缩放因子的低精度矩阵乘法
2. **更低的精度类型**: 支持4位、6位和8位浮点数据类型
3. **按块缩放**: 沿K维度每16或32个元素共享一个缩放因子
4. **OCP兼容**: 支持符合OCP(Open Compute Project)规范的MX数据类型

### 主要数据类型

Blackwell支持多种窄精度数据类型，包括：

| 数据类型 | 位宽 | 描述 |
|----------|------|------|
| float_e2m1_t | 4 | 2位指数、1位尾数的浮点数 |
| float_e2m3_t | 6 | 2位指数、3位尾数的浮点数 |
| float_e3m2_t | 6 | 3位指数、2位尾数的浮点数 |
| float_e4m3_t | 8 | 4位指数、3位尾数的浮点数 |
| float_e5m2_t | 8 | 5位指数、2位尾数的浮点数 |

此外，还支持两种专门用作缩放因子的类型：

| 缩放因子类型 | 位宽 | 描述 |
|-------------|------|------|
| float_ue8m0_t | 8 | 8位指数、0位尾数的无符号浮点数 |
| float_ue4m3_t | 8 | 4位指数、3位尾数的无符号浮点数 |

### Block Scaling机制

Block Scaled GEMMs执行形如$D = C + (A * SFA) * (B * SFB)$的矩阵乘法，其中：

- SFA是A矩阵的缩放因子矩阵
- SFB是B矩阵的缩放因子矩阵
- 每个缩放因子覆盖K维度上的一个数据块（16或32个元素）

具体来说，对于一个$M × K$的A矩阵，会有一个$M × ⌈K/SV⌉$的SFA矩阵，SV是缩放向量大小（Scale Vector Size，通常为16或32）。同样，对于$N × K$的B矩阵，会有一个$N × ⌈K/SV⌉$的SFB矩阵。

## Tensor Memory (TMEM)

Blackwell架构引入了名为Tensor Memory (TMEM)的专用片上内存。TMEM的主要特性包括：

1. **专用于Tensor Core操作**: TMEM是专门为第5代Tensor Core操作设计的内存
2. **规模与组织**: 每个SM有256KB的TMEM，组织为512列和128行的2D结构
3. **动态分配**: 通过`tcgen05.alloc`指令动态分配，以列为单位分配
4. **零寄存器消耗**: UMMA操作的输入和输出直接使用TMEM，不消耗寄存器

TMEM在UMMA（Universal Matrix Multiply-Accumulate）操作中的角色：
- 操作数A可以在TMEM或SMEM中
- 操作数B必须在SMEM中
- 累加器必须在TMEM中

TMEM地址是32位的，其中位31-16表示行ID，位15-0表示列。分配时必须以2的幂次方列数为单位，最小为32列。分配和释放需要通过`tcgen05.alloc`和`tcgen05.dealloc`指令，这些指令必须由同一个warp调用。

## CUTLASS实现

### 数据准备

在[72a_blackwell_nvfp4_bf16_gemm.cu](../examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu)中, 输入类型是NVFP4。

```cpp
template <class F4Type>
struct nv_float4_t
{
  using ScaleFactorType = cutlass::float_ue4m3_t;
  using DataType = F4Type;
};

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
```

除了类型信息外，`SFVectorSize`也是一个关键参数，它定义了每个缩放因子覆盖K维度上的元素数量（通常为16或32）。

在`narrow precision Gemm`中，`SFVectorSize`是在[CollectiveBuilder的OpClassBlockScaledTensorOp](../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L107)特例化版本中自动推导出来的。

```cpp
template <
  class ElementPairA,
  class GmemLayoutATag,
  int AlignmentA,
  class ElementPairB,
  class GmemLayoutBTag,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,        // (MmaAtomShapeM, MmaAtomShapeN, TileK)
  class ClusterShape_MNK,     // Static cluster shape or dynamic (int, int, _1)
  class StageCountType,
  class BuilderScheduleTag
>
struct CollectiveBuilder<
    arch::Sm100,
    arch::OpClassBlockScaledTensorOp, ...>
{
  using ElementSFA = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::sf_type;
  using ElementSFB = typename detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::sf_type;

  using TiledMma = typename cutlass::gemm::collective::detail::TrivialBlockscaledMma<ElementPairA, ElementPairB,
    ElementAccumulator, TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag, is_2sm>::type;

  static constexpr uint32_t SFVectorSize = TiledMma::SFVecSize;
  using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;

  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size
  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF;
  using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
}
```

从上面可以看出，`SFVectorSize`是从自动推导的`TiledMma`中获取到的，而`TiledMma`是从[TrivialBlockscaledMma的特例化版本](../include/cutlass/gemm/collective/builders/sm100_common.inl#L744)得到的。

```cpp
template <
  class ElementPairA,
  class ElementPairB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  UMMA::Major UmmaMajorA,
  UMMA::Major UmmaMajorB,
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
struct TrivialBlockscaledMma<
  ...
  Instr,
  BuilderScheduleTag,
  false /*Is2SM*/> {
    using type = decltype(sm100_make_blockscaled_1sm_trivial_tiled_mma<ElementPairA, ElementPairB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, UmmaMajorA, UmmaMajorB, Instr, BuilderScheduleTag>());
};
```

`TrivialBlockscaledMma`调用[sm100_make_blockscaled_1sm_trivial_tiled_mma](../include/cutlass/gemm/collective/builders/sm100_common.inl#L585)推导出`TiledMma`的类型信息。

```cpp
template <
  ...
  detail::blockscaled::BlockScaledInstr Instr,
  class BuilderScheduleTag
>
constexpr auto
sm100_make_blockscaled_1sm_trivial_tiled_mma() {
  // For MMA_1sm atoms, the MMA's AtomLayout is same as the ClusterShape
  using AtomLayout_MNK = Layout<ClusterShape_MNK>;
  constexpr int M = cute::size<0>(TileShape_MNK{});
  static_assert(M == 128, "Invalid TileShape_M.");

  // Do not allow a tiled MMA N mode > 1, as that is not reasonable.
  constexpr int N = cute::size<1>(TileShape_MNK{});
  static_assert(N == 64 || N == 128 || N == 192 || N == 256, "Invalid TileShape_N.");

  constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
  [[maybe_unused]] constexpr uint32_t SfVectorSizeB = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairB>::SfVectorSize;

  using ElementSF = ElementSFA;

  if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4F6F8) {
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF, M, N, UmmaMajorA, UmmaMajorB>{});
    } else {
      return make_tiled_mma(cute::SM100_MMA_MXF8F6F4_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF, M, N, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else if constexpr (Instr == detail::blockscaled::BlockScaledInstr::MXF4_NVF4) {
    constexpr int SfVectorSize = SfVectorSizeA;
    if constexpr (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ) {
      return make_tiled_mma(cute::SM100_MMA_MXF4NVF4_SS_SPARSE<ElementAMma, ElementBMma, ElementAccumulator, ElementSF, M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    } else {
      return make_tiled_mma(cute::SM100_MMA_MXF4_SS<ElementAMma, ElementBMma, ElementAccumulator, ElementSF, M, N, SfVectorSize, UmmaMajorA, UmmaMajorB>{});
    }
  }
  else {
    static_assert(cutlass::detail::dependent_false<ElementAMma>, "Unsupported configuration for SM100 collective builder.");
  }
}
```

从中我们可以看到，`SfVectorSize`等同于`SfVectorSizeA`, 其定义如下:

```cpp
constexpr uint32_t SfVectorSizeA = detail::blockscaled::blockscaled_type<BuilderScheduleTag, ElementPairA>::SfVectorSize;
```

最终我们定位到[blockscaled_type的nv_float4_t特例化版本](../include/cutlass/gemm/collective/builders/sm1xx_common.inl#L456)，发现`SfVectorSize`的取值是硬编码的。

```cpp
template <class BuilderScheduleTag, class T>
struct blockscaled_type<BuilderScheduleTag, nv_float4_t<T>> {
  using sf_type = cutlass::float_ue4m3_t;
  using data_type = T;
  static constexpr uint32_t SfVectorSize =
    (cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm100, BuilderScheduleTag> ||
     cute::is_base_of_v<KernelScheduleBlockScaledSparseGemmSm120, BuilderScheduleTag>) ? 32 : 16;
};
```

这个设计表明，对于稀疏GEMM操作（使用`KernelScheduleBlockScaledSparseGemm`系列标签），每个缩放因子会覆盖32个元素，而对于常规操作，每个缩放因子覆盖16个元素。

### 内存布局

#### 共享内存布局

`SmemLayoutSFA`和`SmemLayoutSFB`是在[CollectiveMMA](../include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L230)中定义的，而`SmemLayoutAtomSFA`和`SmemLayoutAtomSFB`则是在[CollectiveBuilder](../include/cutlass/gemm/collective/builders/sm100_blockscaled_umma_builder.inl#L209)里定义的：

```cpp
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVectorSize>;
using SmemLayoutAtomSFA = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape_MNK{}));
using SmemLayoutAtomSFB = decltype(Sm1xxBlkScaledConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape_MNK{}));

// 添加pipeline stages维度的最终布局
using SmemLayoutSFA = decltype(make_layout(
  append(shape(SmemLayoutAtomSFA{}), Int<DispatchPolicy::Stages>{}),
  append(stride(SmemLayoutAtomSFA{}), size(filter_zeros(SmemLayoutAtomSFA{})))
));
```

最终我们溯源到`Sm1xxBlockScaledConfig`以及`Sm1xxBlockScaledBasicChunk`，他们均来自于[sm100_blockscaled_layout.hpp](../include/cutlass/detail/sm100_blockscaled_layout.hpp#L49)。

```cpp
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {
  using Blk_MN    = _128;
  using Blk_SF    =   _4;
  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>,
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>,
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};

template<int SFVecSize_>
struct Sm1xxBlockScaledConfig {
  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE static constexpr auto deduce_smem_layoutSFA(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
    constexpr int MMA_NSF = TiledMma::K / SFVecSize;
    // Basic storage block for new Scaling Factor Layouts
    using mnBasicBlockShape  =  Shape<_32,_4>;
    using mnBasicBlockStride = Stride<_16,_4>;
    using kBasicBlockShape  = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
    using kBasicBlockStride = Stride<_0, _1>;

    // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
    // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size
    using Blk_MN    = typename Sm1xxBlkScaledChunk::Blk_MN;
    using Blk_SF    = typename Sm1xxBlkScaledChunk::Blk_SF;
    using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

    constexpr int MMA_M = cute::size<0>(TileShape_MNK{}) / cute::size<0>(typename TiledMma::ThrLayoutVMNK{});
    using mma_SFA_shape  = decltype( make_shape( prepend(Int<MMA_M>{}/Blk_MN{},  mnBasicBlockShape{}),  kBasicBlockShape{}));
    using mma_SFA_stride = decltype(make_stride( prepend(          Blk_Elems{}, mnBasicBlockStride{}), kBasicBlockStride{}));
    using sSFA_shape     = decltype( make_shape( mma_SFA_shape{}, _1{},   make_shape( Blk_SF{}/Int<MMA_NSF>{}, Int<size<2>(TileShape_MNK{}) / SFVecSize / Blk_SF{}>{})));
    using sSFA_stride    = decltype(make_stride(mma_SFA_stride{}, _0{},  make_stride(          Int<MMA_NSF>{},                   Int<MMA_M /Blk_MN{} * Blk_Elems{}>{})));
    using SmemLayoutAtomSFA = decltype(make_layout(sSFA_shape{}, sSFA_stride{}));
    return SmemLayoutAtomSFA{};
  }
}
```

#### 全局内存布局

`layout_SFA`和`layout_SFB`分别通过`tile_atom_to_shape_SFA()`和`tile_atom_to_shape_SFB()`函数设置：

```cpp
using Sm1xxBlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, options.l));
layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, options.l));
```

#### TMEM布局

Blackwell特有的TMEM布局在`init_tmem_tensors`函数中设置：

```cpp
template <class EpilogueTile, bool IsOverlappingAccum = false>
CUTLASS_DEVICE static
auto
init_tmem_tensors(EpilogueTile epi_tile) {
  // ...
  Tensor accumulators = cutlass::detail::make_sm100_accumulator<AccumulatorPipelineStageCount, IsOverlappingAccum>(
      tiled_mma, acc_shape, EpilogueTile{});
  Tensor tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(shape(SmemLayoutAtomSFA{}));
  Tensor tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(shape(SmemLayoutAtomSFB{}));
  // ...
}
```

### TMEM分配与使用

TMEM的分配和使用在CUTLASS中通过以下方式实现：

```cpp
// 实例化分配器
cute::TMEM::Allocator1Sm tmem_allocator{};

// 分配TMEM
tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &shared_storage.tmem_base_ptr);

// 设置TMEM地址
uint32_t tmem_base_ptr = shared_storage.tmem_base_ptr;
collective_mainloop.set_tmem_offsets(tmem_storage, tmem_base_ptr);

// 释放TMEM
tmem_allocator.release_allocation_lock();
tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
```

### TMA（Tensor Memory Accelerator）与数据加载

与Hopper类似，Blackwell也使用TMA进行高效的数据传输。SM100中的TMA实现进一步优化以支持Block Scaling：

1. **构建TMA描述符**：

```cpp
TMA_SFA tma_load_sfa = make_tma_atom_A_sm100<uint16_t>(
    GmemTiledCopySFA{},
    make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}),
    SmemLayoutSFA{}(_,_,_,cute::Int<0>{}),
    TileShape{},
    TiledMma{},
    ClusterLayout_VMNK{});
```

2. **根据大小选择TMA或CP.ASYNC**：

```cpp
static constexpr int ScaleTmaThreshold = 32;
static constexpr bool IsTmaLoadSFA = ScaleMsPerTile >= ScaleTmaThreshold && ScaleNsPerTile < ScaleTmaThreshold;
static constexpr bool IsTmaLoadSFB = ScaleNsPerTile >= ScaleTmaThreshold && ScaleMsPerTile < ScaleTmaThreshold;
```

### Warp特化设计

SM100 Block Scaled GEMM继承了Hopper的warp特化设计并进行进一步优化，将线程块内的工作分配给不同角色的warp：

```cpp
enum class WarpCategory : int32_t {
  MMA          = 0,  // 专注于计算
  Sched        = 1,  // 调度器
  MainloopLoad = 2,  // 加载主循环数据（矩阵A、B及其缩放因子）
  EpilogueLoad = 3,  // 加载epilogue数据
  Epilogue     = 4   // 执行epilogue
};
```

### 缩放因子应用

缩放因子在MMA计算期间应用，通过以下方式：

```cpp
// 在mma函数中
for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
  // (V,M) x (V,N) => (V,M,N)
  cute::gemm(tiled_mma.with(tiled_mma.accumulate_,
                           tCtSFA(_,_,k_block),
                           tCtSFB_mma(_,_,k_block)),
      tCrA(_,_,k_block,read_stage),
      tCrB(_,_,k_block,read_stage),
      accumulators);
  tiled_mma.accumulate_ = UMMA::ScaleOut::One;
}
```

这段代码表明，缩放因子`tCtSFA`和`tCtSFB_mma`被直接传递给`tiled_mma`，由硬件指令自动应用缩放。

### 核心运算流程

Block Scaled GEMM的运算流程可以总结为以下步骤：

1. **初始化**：设置各种布局，分配内存，初始化矩阵和缩放因子
2. **分配TMEM**：为计算申请Tensor Memory空间
3. **加载数据**：使用TMA加载矩阵A、B及其缩放因子SFA、SFB
4. **执行MMA**：在主循环中执行带缩放的矩阵乘法
5. **Epilogue**：处理结果并写回全局内存

其中关键的执行MMA步骤利用了tcgen05.mma.block_scale指令，该指令可以在单个操作中执行乘法、应用缩放因子并累加结果。

## TMEM与Hopper架构的区别

相比Hopper架构，Blackwell的Block Scaled GEMM有以下关键区别：

1. **专用指令**：使用tcgen05.mma.block_scale系列指令而非wgmma指令
2. **更低精度**：支持4位和6位浮点数（Hopper仅支持8位及以上）
3. **性能提升**：与Hopper相比，4位运算可提供4倍性能
4. **TMEM使用**：更高效地利用per-SM Tensor Memory
5. **单线程启动**：与WGMMA不同，UMMA只需一个线程发起，即使使用两个CTA
6. **缩放粒度**：更灵活的缩放因子配置

## 实现优化和性能考量

1. **数据布局优化**：
   ```cpp
   // 为减少内存访问，SMEM布局设计为先沿K模式平铺，再沿MN平铺
   using SmemLayoutA = decltype(UMMA::tile_to_mma_shape(
       SmemLayoutAtomA{},
       append(MmaShapeA_MK{}, Int<DispatchPolicy::Stages>{}),
       cute::conditional_t<cutlass::gemm::detail::is_mn_major<StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
   ```

2. **TMA阈值优化**：
   ```cpp
   // 根据缩放因子大小选择最佳加载方式
   static constexpr int ScaleTmaThreshold = 32;
   ```

3. **异步计算与数据传输**：
   ```cpp
   // 使用双缓冲技术重叠计算与数据传输
   static constexpr bool IsOverlappingAccum = DispatchPolicy::IsOverlappingAccum;
   ```

4. **特殊处理**：
   ```cpp
   // 特殊处理N=192和N=64的情况
   static constexpr int IsCtaN192 = shape<1>(CtaShape_MNK{}) == 192;
   static constexpr int IsCtaN64 = shape<1>(CtaShape_MNK{}) == 64;
   ```

## 历史背景与演进

从历史发展角度看，Blackwell的TMEM和UMMA指令是NVIDIA专用计算资源替代通用资源趋势的延续：

1. **Volta架构**：引入Tensor Core，将GEMM算术操作从通用计算流水线中分离
2. **Ampere架构**：通过异步拷贝指令实现GEMM主循环的真正流水线处理
3. **Hopper架构**：通过异步单线程TMA和warpgroup间寄存器重分配，减少数据移动的寄存器和线程消耗
4. **Blackwell架构**：通过TMEM和UMMA对MMA操作实现了与TMA类似的优化，使其成为单线程、异步操作，不消耗寄存器

这一演进使得寄存器可以主要用于其他任务，如调度和融合epilogue操作，而不是用于存储矩阵数据。

## 结论

NVIDIA Blackwell SM100架构的Block Scaling技术为深度学习提供了极高效的低精度计算能力。CUTLASS通过以下方式充分利用这一特性：

1. 支持4-8位窄精度数据类型及其配套缩放因子
2. 设计高效的内存布局，适配硬件特性
3. 利用warp特化设计分离计算和数据移动
4. 通过TMA和TMEM优化内存访问和计算
5. 为不同数据类型和精度提供灵活配置

通过这些技术，CUTLASS能够显著加速深度学习工作负载，同时保持计算精度，为大规模AI训练和推理提供更高效的硬件利用率。

## References

- [NVIDIA PTX ISA Documentation - Block Scaling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-block-scaling)
- [CUTLASS Blackwell Functionality](../media/docs/cpp/blackwell_functionality.md)
- [CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)

