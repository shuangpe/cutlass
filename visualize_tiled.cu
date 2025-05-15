/* visualize_tiled.cu

Usage:

nvcc visualize_tiled.cu -w \
  -Xcompiler "-Wfatal-errors" \
  -lineinfo \
  -std=c++17 \
  -I/usr/local/cuda/include \
  -I/home/fangwen/shuangpeng/cutlass/include \
  -I/home/fangwen/shuangpeng/cutlass/tools/util/include

./a.out > tiled_mma.tex
pdflatex tiled_mma.tex
realpath tiled_mma.pdf
*/


#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"


using namespace cute;

void print_header() {
  const char* latex_header =
    "\\documentclass{article}\n"
    "\\usepackage[a4paper, margin=0.5cm]{geometry}\n"
    "\\usepackage{adjustbox}\n"
    "\\usepackage{graphicx}\n"
    "\\usepackage{lipsum}\n"
    "\\usepackage{tikz}\n"
    "\n"
    "\\begin{document}\n";
  printf("%s", latex_header);
}


void print_footer() {
  const char* latex_footer = "\\end{document}\n";
  printf("%s", latex_footer);
}

// Copy from mma_atom.hpp
//
// Modified to remove printing header and footder, hence allows printing
// multiple MMAs per TEX file for easier comparisons.
template <class Atom, class TiledThr, class TiledPerm>
void print_mma(const char* name, TiledMMA<Atom, TiledThr, TiledPerm> const& mma) {
  printf("\n\\newpage\n");
  printf("\\begin{verbatim}\n");
  printf("\n%s\n\n", name);
  
  print("ThrLayoutVMNK:  "); print(mma.get_thr_layout_vmnk());  print("\n");
  print("PermutationMNK: "); print(TiledPerm{}); print("\n\n");
  
  print("ThrID:      "); print(typename Atom::ThrID{});      print("\n");
  print("Shape_MNK:  "); print(typename Atom::Shape_MNK{});  print("\n");
  print("LayoutA_TV: "); print(typename Atom::LayoutA_TV{}); print("\n");
  print("LayoutB_TV: "); print(typename Atom::LayoutB_TV{}); print("\n");
  print("LayoutC_TV: "); print(typename Atom::LayoutC_TV{}); print("\n\n");
  printf("\\end{verbatim}\n");

  printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}%");
  printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
         ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");

  auto layout_and_thrid_C = mma.get_layoutC_MN();
  auto layoutC_MN = get<0>(layout_and_thrid_C);
  auto thrID_C    = get<1>(layout_and_thrid_C);

  auto layout_and_thrid_A = mma.get_layoutA_MK();
  auto layoutA_MK = get<0>(layout_and_thrid_A);
  auto thrID_A    = get<1>(layout_and_thrid_A);

  auto layout_and_thrid_B = mma.get_layoutB_NK();
  auto layoutB_NK = get<0>(layout_and_thrid_B);
  auto thrID_B    = get<1>(layout_and_thrid_B);

  auto C = layoutC_MN;
  auto TC = thrID_C;
  auto A = layoutA_MK;
  auto TA = thrID_A;
  auto B = layoutB_NK;
  auto TB = thrID_B;

  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // C starting at 0,0
  for (int m = 0; m < size<0>(C); ++m) {
    for (int n = 0; n < size<1>(C); ++n) {
      int thrid   = C(m,n) % size(TC);
      int val_idx = C(m,n) / size(TC);
      int thr_idx = TC(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, n,
             thr_idx, val_idx);
    }
  }

  // A starting at 0,-size<1>(A)-1
  for (int m = 0; m < size<0>(A); ++m) {
    for (int k = 0; k < size<1>(A); ++k) {
      int thrid   = A(m,k) % size(TA);
      int val_idx = A(m,k) / size(TA);
      int thr_idx = TA(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             m, k-1-size<1>(A),
             thr_idx, val_idx);
    }
  }

  // B starting at -size<1>(B)-1,0
  for (int n = 0; n < size<0>(B); ++n) {
    for (int k = 0; k < size<1>(B); ++k) {
      int thrid   = B(n,k) % size(TB);
      int val_idx = B(n,k) / size(TB);
      int thr_idx = TB(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             k-1-size<1>(B), n,
             thr_idx, val_idx);
    }
  }

  // A labels
  for (int m = 0, k = -1; m < size<0>(A); ++m) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), m);
  }
  for (int k = 0, m = -1; k < size<1>(A); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), k);
  }
  // B labels
  for (int n = 0, k = -1; n < size<0>(B); ++n) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, n);
  }
  for (int k = 0, n = -1; k < size<1>(B); ++k) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, k);
  }

  printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
}


template <class Atom, class... Args>
void print_copy(const char* name, TiledCopy<Atom, Args...> const& copy) {
  using Copy = TiledCopy<Atom, Args...>;

  printf("\n\\newpage\n");

  auto [layoutS_MN, thrID_S] = copy.get_layoutS_MN();
  auto [layoutD_MN, thrID_D] = copy.get_layoutD_MN();
  
  auto S = layoutS_MN;
  auto TS = thrID_S;
  auto D = layoutD_MN;
  auto TD = thrID_D;

  CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

  assert(size<0>(S) == size<0>(D));
  assert(size<1>(S) == size<1>(D));

  printf("\\begin{verbatim}\n");
  printf("\n%s\n\n", name);

  print("  Tiler_MN:       "); print(typename Copy::Tiler_MN{});       print("\n");
  print("  TiledLayout_TV: "); print(typename Copy::TiledLayout_TV{}); print("\n\n");
    
  print("  ThrID:        "); print(typename Atom::ThrID{});        print("\n");
  print("  ValLayoutSrc: "); print(typename Atom::ValLayoutSrc{}); print("\n");
  print("  ValLayoutDst: "); print(typename Atom::ValLayoutDst{}); print("\n");
  print("  ValLayoutRef: "); print(typename Atom::ValLayoutRef{}); print("\n");
  print("  ValueType:    "); print(sizeof_bits<typename Atom::ValType>::value); print("b\n\n");

  printf("      LayoutS: "); print(S);  printf("\n");
  printf("       ThrIDS: "); print(TS); printf("\n");
  printf("      LayoutD: "); print(D);  printf("\n");
  printf("       ThrIDD: "); print(TD); printf("\n");
  printf("\\end{verbatim}\n");

  printf("\\begin{adjustbox}{max height=0.7\\textheight,max width=\\textwidth}%");
  printf("\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
         ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n");
  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // S starting at 0,0
  for (int i = 0; i < size<0>(S); ++i) {
    for (int j = 0; j < size<1>(S); ++j) {
      int thrid   = S(i,j) % size(TS);
      int val_idx = S(i,j) / size(TS);
      int thr_idx = TS(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i, j,
             thr_idx, val_idx);
    }
  }

  // D starting at 0,size<1>(S)+3
  for (int i = 0; i < size<0>(D); ++i) {
    for (int j = 0; j < size<1>(D); ++j) {
      int thrid   = D(i,j) % size(TD);
      int val_idx = D(i,j) / size(TD);
      int thr_idx = TD(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i + size<0>(S) + 3, j,
             thr_idx, val_idx);
    }
  }

  // S Labels
  for (int i = 0, j = -1; i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(S); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }
  // D Labels
  for (int i = 0, j = -1; i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(D); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i + size<0>(S) + 3, j, j);
  }

  printf("\\end{tikzpicture}\n\\end{adjustbox}%\n");
}


void print_layouts_for_mma() {
  using namespace cute;
  using _X = cute::Underscore;
  using MMA_Op = SM70_8x8x4_F32F16F16F32_NT;

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{});
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{}, Layout<Shape<_1,_2>>{});
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{}, Layout<Shape<_2,_1>>{});
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{}, Layout<Shape<_2,_2>>{});
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{}, Layout<Shape<_2,_2>>{},  // AtomLayoutMNK
                                    Tile<Layout<Shape<_2,_1>, Stride<_1,_2>>>{} // PermutationsMNK
    );
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    Layout<Shape<_2,_2>>{},  // AtomLayoutMNK
                                    Tile<Layout<Shape<_1,_2>, Stride<_2,_1>>>{} // PermutationsMNK
    );
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    Layout<Shape<_1,_1,_2>>{},  // AtomLayoutMNK
                                    Tile<Layout<Shape<_1,_1>, Stride<_1,_1>>>{} // PermutationsMNK
    );
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    Layout<Shape<_1,_1>>{},     // AtomLayoutMNK
                                    Tile<Layout<Shape<_1,_1,_2>, Stride<_1,_1,_1>>>{} // PermutationsMNK
    );
    print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  }

  // {
  //   auto tiled_mma = make_tiled_mma(MMA_Op{},
  //                                   Layout<Shape<_2,_2>>{},        // 2x2 layout of atoms (threads)
  //                                   Tile<Layout<Shape <_4,_2>,     // Permutation in M
  //                                               Stride<_1,_8>>>{});
  //   print_mma("SM70_8x8x4_F32F16F16F32_NT", tiled_mma);
  // }

#if 0
  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    Layout<Shape<_2,_2>>{},      // 2x2 layout of atoms (threads)
                                    Tile<Layout<Shape <_4,_2>,   // Permutation in M
                                                Stride<_1,_8>>>{});

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    composition(Swizzle<1,0,-1>{},
                                                Layout<Shape <Shape <_2,_2>,Shape <_2,_2>>,
                                                       Stride<Stride<_1,_4>,Stride<_2,_8>>>{}),  // 4x4 layout of atoms (threads)
                                    Tile<Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>,              // Permutation in M
                                         Layout<Shape<_4,_2,_2>,Stride<_1,_32,_8>>>{});          // Permutation in N

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(MMA_Op{},
                                    Layout<Shape<_2,_2>>{},      // 2x2 tile of atoms (threads)
                                    Tile<Layout<Shape <_4,_2>,
                                                Stride<_1,_16>>, // Permutation of M
                                         Layout<_1,
                                                _1>>{});         // Permutation of N

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(SM75_16x8x8_F32F16F16F32_TN{});

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{}, // HMMA.16816 warp-wide MmaAtom. 32 threads, 1 warp. Each thread owns 4 output values.
                                    Layout<Shape<_2,_2>>{},   // Tile in Threads/Warp
                                    Layout<Shape<_1,_1>>{});  // Tile in Output Val per thread.

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(SM80_8x8x4_F64F64F64F64_TN{},
                                    Layout<Shape<_2,_2>>{},  // Layout in Thr
                                    Layout<Shape<_2,_2>>{},  // Layout in Val
                                    Tile<Layout<_2,          // Permutation on M
                                                _16>,
                                         Layout<_2,          // Permutation on N
                                                _16>>{});

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{});

    print_latex(tiled_mma);
  }
#endif

#if 0
  {
    auto tiled_mma = make_tiled_mma(SM90_64x32x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

    print_latex(tiled_mma);
  }
#endif
}

void print_layouts_for_sm80_cp_async_cachealways() {
  using ElementFlt = float;
  using Atom = Copy_Atom<UniversalCopy<ElementFlt>, ElementFlt>;
  
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      Layout<Shape < _1, Shape<_4,_8>>, Stride<_0, Stride<_1, _4>>>{},
      Layout<Shape < _1, _1>>{}                   // ValLayout (m,n) -> val_idx
    );
    // print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
    
    auto tensor = make_tensor<ElementFlt>(make_layout(Shape<_1,_128>{}, GenRowMajor{}));
    print("Tensor:"); print(tensor); print("\n");
    
    for (int i = 0; i < 3; ++i) {
      auto thr_copy = tiled_copy.get_slice(i);
      auto sliced = thr_copy.partition_S(tensor);
      print("Sliced:"); print(sliced); print("\n");
    }
  }

  #if 0
    
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      make_layout(Shape<_4,_2>{}, GenRowMajor{}), // ThrLayout (m,n) -> thr_idx
      Layout<Shape < _1, _4>>{}                   // ValLayout (m,n) -> val_idx
    );
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }
  
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      make_layout(Shape<_4,_2>{}, GenColMajor{}), // ThrLayout (m,n) -> thr_idx
      Layout<Shape < _1, _4>>{}                   // ValLayout (m,n) -> val_idx
    );
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }
  
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      make_layout(Shape<_4,_2>{}, GenRowMajor{}), // ThrLayout (m,n) -> thr_idx
      make_layout(Shape<_2,_2>{}, GenRowMajor{})                  // ValLayout (m,n) -> val_idx
    );
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }
  
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      make_layout(Shape<_4,_2>{}, GenRowMajor{}), // ThrLayout (m,n) -> thr_idx
      make_layout(Shape<_2,_2>{}, GenColMajor{})                  // ValLayout (m,n) -> val_idx
    );
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }
  
  {
    auto tiled_copy = make_tiled_copy(Atom{},
      make_layout(Shape<_4,_2>{}, GenColMajor{}), // ThrLayout (m,n) -> thr_idx
      Layout<Shape < _1, _4>>{}                   // ValLayout (m,n) -> val_idx
    );
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_32, _1>>,
      Layout<Shape<_4, _8>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_2, _1>>,
      Layout<Shape<_8, _4>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_16, _1>>,
      Layout<Shape<_8, _8>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_32, _1>>,
      Layout<Shape<_8, _8>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_4, _4>>,
      Layout<Shape<_8, _8>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }

  {
    using ST = float;
    using DT = float;
    using Copy_Atom_Arch = Copy_Atom<Copy_Traits<SM80_CP_ASYNC_CACHEALWAYS<ST, DT>>, DT>;
    auto tiled_copy = TiledCopy<
      Copy_Atom_Arch,
      Layout<Shape<_4, _1>>,
      Layout<Shape<_8, _32>, Stride<_32, _32>>
    >{};
    print_copy("SM80_CP_ASYNC_CACHEALWAYS", tiled_copy);
  }
#endif
}


int main() {
  print_header();

  // print_layouts_for_mma();
  print_layouts_for_sm80_cp_async_cachealways();

  print_footer();
  return 0;
}