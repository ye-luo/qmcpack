//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2022 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "syclBLAS.hpp"
#include "oneapi/mkl/blas.hpp"

namespace qmcplusplus
{
namespace syclBLAS
{
inline oneapi::mkl::transpose convertTransEnum(char trans)
{
  return trans == 'T' ? oneapi::mkl::transpose::trans : oneapi::mkl::transpose::nontrans;
}

template<typename T>
sycl::event gemv(sycl::queue& handle,
                 const char trans,
                 const int m,
                 const int n,
                 const T alpha,
                 const T* const A,
                 const int lda,
                 const T* const x,
                 const int incx,
                 const T beta,
                 T* const y,
                 const int incy,
                 const std::vector<sycl::event>& events)
{
  return oneapi::mkl::blas::gemv(handle, convertTransEnum(trans), m, n, alpha, A, lda, x, incx, beta, y, incy, events);
}

template sycl::event gemv(sycl::queue& handle,
                          const char trans,
                          const int m,
                          const int n,
                          const double alpha,
                          const double* const A,
                          const int lda,
                          const double* const x,
                          const int incx,
                          const double beta,
                          double* const y,
                          const int incy,
                          const std::vector<sycl::event>& events);

template sycl::event gemv(sycl::queue& handle,
                          const char trans,
                          const int m,
                          const int n,
                          const float alpha,
                          const float* const A,
                          const int lda,
                          const float* const x,
                          const int incx,
                          const float beta,
                          float* const y,
                          const int incy,
                          const std::vector<sycl::event>& events);

template sycl::event gemv(sycl::queue& handle,
                          const char trans,
                          const int m,
                          const int n,
                          const std::complex<double> alpha,
                          const std::complex<double>* const A,
                          const int lda,
                          const std::complex<double>* const x,
                          const int incx,
                          const std::complex<double> beta,
                          std::complex<double>* const y,
                          const int incy,
                          const std::vector<sycl::event>& events);

template sycl::event gemv(sycl::queue& handle,
                          const char trans,
                          const int m,
                          const int n,
                          const std::complex<float> alpha,
                          const std::complex<float>* const A,
                          const int lda,
                          const std::complex<float>* const x,
                          const int incx,
                          const std::complex<float> beta,
                          std::complex<float>* const y,
                          const int incy,
                          const std::vector<sycl::event>& events);

template<typename T>
sycl::event gemv_batched(sycl::queue&   handle,
                         const char          trans,
                         const syclBLAS_int  m,
                         const syclBLAS_int  n,
                         const T*            alpha,
                         const T**           A,
                         const syclBLAS_int  lda,
                         const T**           X,
                         const syclBLAS_int  incx,
                         const T*            beta,
                         T**                 Y,
                         const syclBLAS_int  incy,
                         const syclBLAS_int  batch_count,
                         const std::vector<sycl::event> &events)
{
  oneapi::mkl::transpose trA = convertTransEnum(trans);
  //using group api: only one group
  return oneapi::mkl::blas::gemv_batch(handle, &trA, &m, &n, alpha, A, &lda,
                                       X,&incx, beta, Y, &incy,1,&batch_count, events);
}

template sycl::event gemv_batched(sycl::queue&   handle,
                                  const char          trans,
                                  const syclBLAS_int  m,
                                  const syclBLAS_int  n,
                                  const float*        alpha,
                                  const float**       A,
                                  const syclBLAS_int  lda,
                                  const float**       X,
                                  const syclBLAS_int  incx,
                                  const float*        beta,
                                  float**             Y,
                                  const syclBLAS_int  incy,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

template sycl::event gemv_batched(sycl::queue&   handle,
                                  const char          trans,
                                  const syclBLAS_int  m,
                                  const syclBLAS_int  n,
                                  const double*       alpha,
                                  const double**      A,
                                  const syclBLAS_int  lda,
                                  const double**      X,
                                  const syclBLAS_int  incx,
                                  const double*       beta,
                                  double**            Y,
                                  const syclBLAS_int  incy,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

template<typename T>
sycl::event gemm(sycl::queue& handle,
                 const char tA,
                 const char tB,
                 const int m,
                 const int n,
                 const int k,
                 const T alpha,
                 const T* A,
                 const int lda,
                 const T* B,
                 const int ldb,
                 const T beta,
                 T* C,
                 const int ldc,
                 const std::vector<sycl::event>& events)
{
  return oneapi::mkl::blas::gemm(handle, convertTransEnum(tA), convertTransEnum(tB), m, n, k, alpha, A, lda, B, ldb,
                                 beta, C, ldc, events);
}


template sycl::event gemm(sycl::queue& handle,
                          const char tA,
                          const char tB,
                          const int m,
                          const int n,
                          const int k,
                          const float alpha,
                          const float* const A,
                          const int lda,
                          const float* const B,
                          const int ldb,
                          const float beta,
                          float* const C,
                          const int ldc,
                          const std::vector<sycl::event>& events);

template sycl::event gemm(sycl::queue& handle,
                          const char tA,
                          const char tB,
                          const int m,
                          const int n,
                          const int k,
                          const double alpha,
                          const double* const A,
                          const int lda,
                          const double* const B,
                          const int ldb,
                          const double beta,
                          double* const C,
                          const int ldc,
                          const std::vector<sycl::event>& events);

template sycl::event gemm(sycl::queue& handle,
                          const char tA,
                          const char tB,
                          const int m,
                          const int n,
                          const int k,
                          const std::complex<float> alpha,
                          const std::complex<float>* const A,
                          const int lda,
                          const std::complex<float>* const B,
                          const int ldb,
                          const std::complex<float> beta,
                          std::complex<float>* const C,
                          const int ldc,
                          const std::vector<sycl::event>& events);

template sycl::event gemm(sycl::queue& handle,
                          const char tA,
                          const char tB,
                          const int m,
                          const int n,
                          const int k,
                          const std::complex<double> alpha,
                          const std::complex<double>* const A,
                          const int lda,
                          const std::complex<double>* const B,
                          const int ldb,
                          const std::complex<double> beta,
                          std::complex<double>* const C,
                          const int ldc,
                          const std::vector<sycl::event>& events);

template<typename T>
sycl::event gemm_batched(sycl::queue&   handle,
                         const char transA,
                         const char transB,
                         const syclBLAS_int  m,
                         const syclBLAS_int  n,
                         const syclBLAS_int  k,
                         const T*            alpha,
                         const T**           A,
                         const syclBLAS_int  lda,
                         const T**           B,
                         const syclBLAS_int  ldb,
                         const T*            beta,
                         T**                 C,
                         const syclBLAS_int  ldc,
                         const syclBLAS_int  batch_count,
                         const std::vector<sycl::event> &events)
{
  oneapi::mkl::transpose trA = convertTransEnum(transA);
  oneapi::mkl::transpose trB = convertTransEnum(transB);
  return oneapi::mkl::blas::gemm_batch(handle, &trA, &trB, &m, &n, &k, alpha, A, &lda, B,&ldb, beta, C, &ldc,1,&batch_count, events);
}

template sycl::event gemm_batched(sycl::queue&        handle,
                                  const char          transA,
                                  const char          transB,
                                  const syclBLAS_int  m,
                                  const syclBLAS_int  n,
                                  const syclBLAS_int  k,
                                  const float*        alpha,
                                  const float**       A,
                                  const syclBLAS_int  lda,
                                  const float**       B,
                                  const syclBLAS_int  ldb,
                                  const float*        beta,
                                  float**             C,
                                  const syclBLAS_int  ldc,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

template sycl::event gemm_batched(sycl::queue&        handle,
                                  const char          transA,
                                  const char          transB,
                                  const syclBLAS_int  m,
                                  const syclBLAS_int  n,
                                  const syclBLAS_int  k,
                                  const double*       alpha,
                                  const double**      A,
                                  const syclBLAS_int  lda,
                                  const double**      B,
                                  const syclBLAS_int  ldb,
                                  const double*       beta,
                                  double**            C,
                                  const syclBLAS_int  ldc,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

template<typename T>
sycl::event ger_batched(sycl::queue& handle,
                        const int       m,
                        const int       n,
                        const T*        alpha,
                        const T**       x,
                        const int       incx,
                        const T**       y,
                        const int       incy,
                        T**             A,
                        const int       lda,
                        const size_t    batch_count)
{
#if 0
  const size_t COLBS=16;
  const size_t m_max=((m+COLBS-1)/COLBS)*COLBS;
  const size_t n_max=((n+COLBS-1)/COLBS)*COLBS;
  return handle.parallel_for(
      sycl::nd_range<3>{{batch_count,m_max,n_max},{1,COLBS,COLBS}},
          [=](sycl::nd_item<3> item) {
          const unsigned batch = item.get_global_id(0);
          const unsigned y_g = item.get_global_id(1);
          const unsigned x_g = item.get_global_id(2);
          if(y_g<m && x_g<n)
             A[batch][y_g*lda + x_g] += alpha[batch]*x[batch][x_g]*y[batch][y_g];
          });
#endif
  const size_t tile_size = 32;
  const size_t block_rows = 8;
  const size_t n_tiles = ((m + tile_size - 1) / tile_size);

  return handle.parallel_for(
      sycl::nd_range<3>{{batch_count, n_tiles*block_rows, n_tiles*tile_size}, {1, block_rows, tile_size}},
      [=](sycl::nd_item<3> item) {
      const unsigned batch  = item.get_group(0);
      const unsigned thX    = item.get_local_id(2); 
      const unsigned thY    = item.get_local_id(1); 
      const unsigned column = item.get_group(2) * tile_size + thX;
      const unsigned row    = item.get_group(1) * tile_size + thY;
      if(column<n)
      {
        const T alphaX  = alpha[batch] * x[batch][column];
        for (unsigned j = 0; j < tile_size; j += block_rows)
           if(row+j < m) A[batch][(row+j)*lda + column] += alphaX * y[batch][row + j];
      }
  });
}

template sycl::event ger_batched(sycl::queue& handle,
                        const int       m,
                        const int       n,
                        const float*    alpha,
                        const float**   x,
                        const int       incx,
                        const float**   y,
                        const int       incy,
                        float**         A,
                        const int       lda,
                        const size_t    batch_count);

template sycl::event ger_batched(sycl::queue& handle,
                        const int       m,
                        const int       n,
                        const double*   alpha,
                        const double**  x,
                        const int       incx,
                        const double**  y,
                        const int       incy,
                        double**        A,
                        const int       lda,
                        const size_t    batch_count);

//transpose
template<typename T1, typename T2>
sycl::event transpose_base(sycl::queue& q,
                      const T1* restrict in,
                      int m,
                      int lda,
                      T2* restrict out,
                      int n,
                      int ldb,
                      const std::vector<sycl::event>& events)
{
  constexpr size_t tile_size = 16;
  const size_t m_max         = ((m + tile_size - 1) / tile_size) * tile_size;
  const size_t n_max         = ((n + tile_size - 1) / tile_size) * tile_size;

  return q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(events);

#if defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20220928
    sycl::local_accessor<T2, 2> tile(sycl::range<2>(tile_size, tile_size + 1), cgh);
#else
    sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local> tile(sycl::range<2>(tile_size, tile_size + 1), cgh);
#endif

    cgh.parallel_for(sycl::nd_range<2>{{m_max, n_max}, {tile_size, tile_size}}, [=](sycl::nd_item<2> item) {
      unsigned x   = item.get_global_id(1);
      unsigned y   = item.get_global_id(0);
      unsigned xth = item.get_local_id(1);
      unsigned yth = item.get_local_id(0);

      if (x < n && y < m)
        tile[yth][xth] = in[(y)*lda + x];
      item.barrier(sycl::access::fence_space::local_space);

      x = item.get_group(0) * tile_size + xth;
      y = item.get_group(1) * tile_size + yth;
      if (x < m && y < n)
        out[(y)*ldb + x] = tile[xth][yth];
    });
  });
}

template<typename T1, typename T2>
sycl::event transpose(sycl::queue& q,
                      const T1* restrict in,
                      int m,
                      int lda,
                      T2* restrict out,
                      int n,
                      int ldb,
                      const std::vector<sycl::event>& events)
{ 
  const size_t tile_size = 32;
  const size_t block_rows = 8;
  const size_t n_tiles = ((m + tile_size - 1) / tile_size);

  return q.submit([&](sycl::handler& cgh) {
#if defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20220928
      sycl::local_accessor<T2, 2> tile(sycl::range<2>(tile_size,tile_size+1), cgh);
#else
      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local>
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);
#endif
      cgh.parallel_for(sycl::nd_range<2>{{n_tiles*block_rows, n_tiles*tile_size}, {block_rows, tile_size}},
        [=](sycl::nd_item<2> item) {
        const unsigned thX = item.get_local_id(1); //threadIdx.x
        const unsigned thY = item.get_local_id(0); //threadIdx.y
        unsigned column = item.get_group(1) * tile_size + thX; //item.get_global_id(1);
        unsigned row    = item.get_group(0) * tile_size + thY;

	for (unsigned j = 0; j < tile_size; j += block_rows)
          tile[thY+j][thX] = in[(row+j)*lda + column];

        item.barrier(sycl::access::fence_space::local_space);

        column = item.get_group(0)*tile_size + thX;
        if(column<n)
        {
          row = item.get_group(1)*tile_size + thY;
          for (unsigned j = 0; j < tile_size; j += block_rows)
          if(row+j < m) out[(row + j)*ldb + column] = tile[thX][thY + j];
        }
    });
  });
}

template sycl::event transpose(sycl::queue& q,
                               const float* restrict in,
                               int m,
                               int lda,
                               double* restrict out,
                               int n,
                               int ldb,
                               const std::vector<sycl::event>& events);

template sycl::event transpose(sycl::queue& q,
                               const double* restrict in,
                               int m,
                               int lda,
                               double* restrict out,
                               int n,
                               int ldb,
                               const std::vector<sycl::event>& events);

template sycl::event transpose(sycl::queue& q,
                               const std::complex<float>* restrict in,
                               int m,
                               int lda,
                               std::complex<double>* restrict out,
                               int n,
                               int ldb,
                               const std::vector<sycl::event>& events);

template sycl::event transpose(sycl::queue& q,
                               const std::complex<double>* restrict in,
                               int m,
                               int lda,
                               std::complex<double>* restrict out,
                               int n,
                               int ldb,
                               const std::vector<sycl::event>& events);

template<typename T1, typename T2>
sycl::event transpose_batched(sycl::queue& q,
                              const T1** in_batch,
                              int m,
                              int lda,
                              T2* out_batch,
                              int n,
                              int ldb,
                              int batch_count,
                              const std::vector<sycl::event>& events)
{ 
  const size_t tile_size = 32;
  const size_t block_rows = 8;
  const size_t n_tiles = ((m + tile_size - 1) / tile_size);

  return q.submit([&](sycl::handler& cgh) {
#if defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20220928
      sycl::local_accessor<T2, 2> tile(sycl::range<2>(tile_size,tile_size+1), cgh);
#else
      sycl::accessor<T2, 2, sycl::access::mode::write, sycl::access::target::local>
      tile(sycl::range<2>(tile_size,tile_size+1), cgh);
#endif
      cgh.parallel_for(sycl::nd_range<3>{{static_cast<size_t>(batch_count), n_tiles*block_rows, n_tiles*tile_size}, 
            {1,block_rows, tile_size}},
        [=](sycl::nd_item<3> item) {

        const unsigned iw  = item.get_group(0);
        const unsigned thX = item.get_local_id(2); //threadIdx.x
        const unsigned thY = item.get_local_id(1); //threadIdx.y
        unsigned column = item.get_group(2) * tile_size + thX; //item.get_global_id(1);
        unsigned row    = item.get_group(1) * tile_size + thY;

        const T1* restrict in = in_batch[iw];
	for (unsigned j = 0; j < tile_size; j += block_rows)
          tile[thY+j][thX] = in[(row+j)*lda + column];

        item.barrier(sycl::access::fence_space::local_space);

        //T2* restrict out = out_batch[iw];
        T2* restrict out = out_batch + iw * n * ldb;
        column = item.get_group(1)*tile_size + thX;
        if(column<n)
        {
          row = item.get_group(2)*tile_size + thY;
          for (unsigned j = 0; j < tile_size; j += block_rows)
          if(row+j < m) out[(row + j)*ldb + column] = tile[thX][thY + j];
        }
    });
  });
}

template sycl::event transpose_batched(sycl::queue& q,
                                       const float** in_batch,
                                       int m,
                                       int lda,
                                       double* out_batch,
                                       int n,
                                       int ldb,
                                       int batch_count,
                                       const std::vector<sycl::event>& events);

template sycl::event transpose_batched(sycl::queue& q,
                                       const double** in_batch,
                                       int m,
                                       int lda,
                                       double* out_batch,
                                       int n,
                                       int ldb,
                                       int batch_count,
                                       const std::vector<sycl::event>& events);

//copy_n for mixed precision
template<typename T1, typename T2>
sycl::event copy_n(sycl::queue& aq,
                   const T1* restrict VA,
                   size_t array_size,
                   T2* restrict VC,
                   const std::vector<sycl::event>& events)
{
  constexpr size_t tile_size = 64;
  const size_t a_max         = ((array_size + tile_size - 1) / tile_size) * tile_size;
  return aq.parallel_for(sycl::range<1>{a_max}, events, [=](sycl::id<1> id) {
    if (id < array_size)
      VC[id] = static_cast<T2>(VA[id]);
  });
}

template sycl::event copy_n(sycl::queue& aq,
                            const double* restrict VA,
                            size_t array_size,
                            float* restrict VC,
                            const std::vector<sycl::event>& events);

template sycl::event copy_n(sycl::queue& aq,
                            const std::complex<double>* restrict VA,
                            size_t array_size,
                            std::complex<float>* restrict VC,
                            const std::vector<sycl::event>& events);

template sycl::event copy_n(sycl::queue& aq,
                            const double* restrict VA,
                            size_t array_size,
                            double* restrict VC,
                            const std::vector<sycl::event>& events);

template sycl::event copy_n(sycl::queue& aq,
                            const std::complex<double>* restrict VA,
                            size_t array_size,
                            std::complex<double>* restrict VC,
                            const std::vector<sycl::event>& events);


template<typename T>
sycl::event copy_batched(sycl::queue&   handle,
                         const syclBLAS_int  m,
                         const T**           X,
                         const syclBLAS_int  incx,
                         T**                 Y,
                         const syclBLAS_int  incy,
                         const syclBLAS_int  batch_count,
                         const std::vector<sycl::event> &events)
{
  return oneapi::mkl::blas::copy_batch(handle, &m, X, &incx, Y, &incy, 1, &batch_count, events);
}

template sycl::event copy_batched(sycl::queue&   handle,
                                  const syclBLAS_int  m,
                                  const float**       X,
                                  const syclBLAS_int  incx,
                                  float**             Y,
                                  const syclBLAS_int  incy,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

template sycl::event copy_batched(sycl::queue&   handle,
                                  const syclBLAS_int  m,
                                  const double**       X,
                                  const syclBLAS_int  incx,
                                  double**             Y,
                                  const syclBLAS_int  incy,
                                  const syclBLAS_int  batch_count,
                                  const std::vector<sycl::event> &events);

} // namespace syclBLAS

} // namespace qmcplusplus
