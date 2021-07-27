//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////

#include "catch.hpp"

#include <memory>
#include <vector>
#include <iostream>
#include "OMPTarget/OMPallocator.hpp"
#include "OMPTarget/ompBLAS.hpp"
#include <OhmmsPETE/OhmmsVector.h>
#include <OhmmsPETE/OhmmsMatrix.h>
#include <CPU/BLAS.hpp>

namespace qmcplusplus
{

template<typename T>
void test_gemv(const int N, const char trans)
{
  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  ompBLAS::ompBLAS_handle handle;

  vec_t A(N);    // Input vector
  mat_t B(N, N); // Input matrix
  vec_t C(N);    // Result vector ompBLAS
  vec_t D(N);    // Result vector BLAS

  for (int dim = 1; dim <= N; dim++)
  {
    handle = dim;
    for (int i = 0; i < dim; i++)
    {
      A[i] = i;
      for (int j = 0; j < dim; j++)
        B.data()[i * dim + j] = i + j;
    }

    A.updateTo();
    B.updateTo();

    ompBLAS::gemv(handle, trans, dim, dim, 1.0, B.device_data(), dim, A.device_data(), 1, 0.0, C.device_data(),
                  1); // tests omp gemv
    if(trans == 'T')
      {
	BLAS::gemv_trans(dim, dim, B.data(), A.data(), D.data());
      } else
      {
	BLAS::gemv(dim, dim, B.data(), A.data(), D.data());
      }
    
    C.updateFrom();

    bool are_same = true;
    int index     = 0;
    do
    {
      are_same = C[index] == D[index];
      CHECK(C[index] == D[index]);
      index++;
    } while (are_same == true && index < dim);
  }
}

template<typename T>
void test_gemv_batched(const int N, const char trans, const int batch_count)
{

  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  ompBLAS::ompBLAS_handle handle;

  // Create input vector
  std::vector<vec_t> As;
  Vector<const T*, OMPallocator<const T*>> Aptrs;

  // Create input matrix
  std::vector<mat_t> Bs;
  Vector<const T*, OMPallocator<const T*>> Bptrs;

  // Create output vector (ompBLAS)
  std::vector<vec_t> Cs;
  Vector<T*, OMPallocator<T*>> Cptrs;

  // Create output vector (BLAS)
  std::vector<vec_t> Ds;
  Vector<T*, OMPallocator<T*>> Dptrs;

  // Change N to batch size?
  Aptrs.resize(N);
  Bptrs.resize(N * N);
  Cptrs.resize(N);
  Dptrs.resize(N);
  for (int dim = 1; dim <= N; dim++)
    {
      handle = dim;

      // Fill data
      for(int batch = 0; batch < dim; batch++)
	{ 
	  As[batch].resize(dim);
	  Aptrs[batch] = As[batch].device_data();

	  Bs[batch].resize(dim, dim);
	  Bptrs[batch] = Bs[batch].device_data();

	  Cs[batch].resize(dim);
	  Cptrs[batch] = Cs[batch].device_data();
	  
          Ds[batch].resize(dim);
	  Dptrs[batch] = Ds[batch].device_data();
	  
	  for (int i = 0; i < dim; i++)
	    {
	      As[batch][i] = i;
	      for (int j = 0; j < dim; j++)
		Bs[batch].device_data()[i * dim + j] = i + j;
	    }
	  As[batch].updateTo();
	  Bs[batch].updateTo();
	}
      Aptrs.updateTo();
      Bptrs.updateTo();
      Cptrs.updateTo();
      
      // run tests here 
      Vector<T, OMPallocator<T>> alpha;
      alpha.resize(N, T(1));
      Vector<T, OMPallocator<T>> beta;
      beta.resize(N, T(1));

      ompBLAS::gemv_batched(handle, trans, dim, dim, alpha.data(), Bptrs.device_data(), dim, Aptrs.device_data(), 1, beta.data(), Cptrs.device_data(), 1, 13);
      
      for(int batch = 0; batch < dim; batch++)
	{
	  if(trans == 'T')
	    {
	      BLAS::gemv_trans(dim, dim, Bptrs[batch], Aptrs[batch], Dptrs[batch]);
	    } else
	    {
	      BLAS::gemv(dim, dim, Bptrs[batch], Aptrs[batch], Dptrs[batch]);
	    }
	}

      Cptrs.updateFrom();
      
      bool are_same = true;
      int index     = 0;
      for(int batch = 1; batch <= dim; batch++)
	{
	  do
	    {
	      are_same = Cptrs[batch][index] == Dptrs[batch][index];
	      CHECK(are_same);
	      index++;
	    } while (are_same == true && index < dim);
	}
    }
  
}
  
TEST_CASE("OmpBLAS gemv", "[OMP]")
{

  const int N = 100;
  
  // NOTRNS NOT IMPL
  /*
  std::cout << "Testing NOTRANS gemv" << std::endl;
  test_gemv<float>(N, 'N');
  test_gemv<double>(N, 'N');
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>>(N, 'N');
  test_gemv<std::complex<double>>(N, 'N');
#endif
  */

  /*
  std::cout << "Testing TRANS gemv" << std::endl;
  test_gemv<float>(N, 'T');
  test_gemv<double>(N, 'T');
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>>(N, 'T');
  test_gemv<std::complex<double>>(N, 'T');
#endif
  */
  std::cout << "Testing TRANS gemv_batched" << std::endl;
  test_gemv_batched<float>(N, 'T', 13);
  
}
} // namespace qmcplusplus
