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
void test_gemv(const int M, const int N, const char trans)
{
  using vec_t = Vector<T, OMPallocator<T>>;
  using mat_t = Matrix<T, OMPallocator<T>>;

  ompBLAS::ompBLAS_handle handle;

  vec_t A(N);    // Input vector
  mat_t B(M, N); // Input matrix
  vec_t C(N);    // Result vector ompBLAS
  vec_t D(N);    // Result vector BLAS  
  
  // Fill data
  for(int i = 0; i < N; i++)
    {
      A[i] = i;
      for(int j = 0; j < M; j++)
	{
	  B.data()[j * N + i] = i + j * 2;
	}
    }

  // Fill C and D with 0
  for(int i = 0; i < M; i++)
    {
      C[i] = T(0);
      D[i] = T(0);
    }
  
  A.updateTo();
  B.updateTo();

  std::cout << "A[";
  for(int i = 0; i < N; i++)
    std::cout << A[i] << ", ";
  std::cout << "]" << std::endl;

  std::cout << "B[";
  for(int i = 0; i < N * M; i++)
    std::cout << B.data()[i] << ", ";
  std::cout << "]" << std::endl;

  std::cout << "C[";
  for(int i = 0; i < M; i++)
    std::cout << C[i] << ", ";
  std::cout << "]" << std::endl;

  std::cout << "D[";
  for(int i = 0; i < M; i++)
    std::cout << D[i] << ", ";
  std::cout << "]" << std::endl;
  
  T alpha = T(1);
  T beta = T(0);
  // ompBLAS::gemv(handle, trans, M, N, alpha, matrix, LDA = M, vector, incx, beta, result, incy);
  ompBLAS::gemv(handle, trans, M, N, alpha, B.device_data(), M, A.device_data(), 1, beta, C.device_data(), 1);

  // BLAS::gemv(N, M, matrix, vector, result)
  if(trans == 'T')                                                                                                                                                               
      {                                                                                                                                                                           
        BLAS::gemv_trans(N, M, B.data(), A.data(), D.data()); 
      } else
    {
      BLAS::gemv(N, M, B.data(), A.data(), D.data());                                                                                                                        
    }                                                                                                                                                                            
  
  C.updateFrom();
  
  std::cout << "C[";
  for(int i = 0; i < M; i++)
    std::cout << C.data()[i] << ", ";
  std:: cout << "]" << std::endl;

  std::cout << "D[";
  for(int i = 0; i < M; i++)
    std::cout << D.data()[i]<< ", ";
  std::	cout <<	"]" << std::endl;
  
  bool are_same = true;                                                                                                                                                          
  int index     = 0;                                                                                                                                                             
  do                                                                                                                                                                             
    {                                                                                                                                                                             
      are_same = C[index] == D[index];                                                                                                                                            
      CHECK(C[index] == D[index]);                                                                                                                                                
      index++;                                                                                                                                                                    
    } while (are_same == true && index < M);                                          
           
  // Original test for NxN only
  /*
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
  */
}

template<typename T>
void test_gemv_batched(const int M, const int N,  const char trans, const int batch_count)
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
  
  // Resize pointer vectors
  Aptrs.resize(batch_count);
  Bptrs.resize(batch_count);
  Cptrs.resize(batch_count);
  Dptrs.resize(batch_count);

  // Resize data vectors
  As.resize(batch_count);
  Bs.resize(batch_count);
  Cs.resize(batch_count);
  Ds.resize(batch_count);
  
  // Fill data
  for(int batch = 0; batch < batch_count; batch++)
    {
      
      handle = batch;
      
      As[batch].resize(N);
      Aptrs[batch] = As[batch].device_data();
      
      Bs[batch].resize(M, N);
      Bptrs[batch] = Bs[batch].device_data();
      
      Cs[batch].resize(M);
      Cptrs[batch] = Cs[batch].device_data();
      
      Ds[batch].resize(M);
      Dptrs[batch] = Ds[batch].data();
      
      for (int i = 0; i < N; i++)
	{
	  As[batch][i] = i;
	  for (int j = 0; j < M; j++)
	    Bs[batch].data()[i + N * j] = i + j * 2;
	}
      
      As[batch].updateTo();
      Bs[batch].updateTo();
    }
  
  Aptrs.updateTo();
  Bptrs.updateTo();
  Cptrs.updateTo();
  
  // Run tests 
  Vector<T, OMPallocator<T>> alpha;
  alpha.resize(batch_count);
  Vector<T, OMPallocator<T>> beta;
  beta.resize(batch_count);

  for(int batch = 0; batch < batch_count; batch++)
    {
      alpha[batch] = T(1);
      beta[batch] = T(1);
    }
  
  alpha.updateTo();
  beta.updateTo();


  ompBLAS::gemv_batched(handle, trans, M, N, alpha.device_data(), Bptrs.device_data(), M, Aptrs.device_data(), 1, beta.device_data(), Cptrs.device_data(), 1, batch_count);

  
  for(int batch = 0; batch < batch_count; batch++)
    {
      if(trans == 'T')
	{
	  BLAS::gemv_trans(N, M, Bs[batch].data(), As[batch].data(), Ds[batch].data());
	} else
	{
	  BLAS::gemv(N, M, Bs[batch].data(), As[batch].data(), Ds[batch].data());
	}
    }
  
  for(int batch = 0; batch < batch_count; batch++)
    {
      Cs[batch].updateFrom();
    }

  // Check results
  for(int batch = 0; batch < batch_count; batch++)
    {
      bool are_same = true;
      int index     = 0;
      do
	{
	  are_same = Cs[batch][index] == Ds[batch][index];
	  CHECK(are_same);
	  index++;
	} while (are_same == true && index < M);
    } 
}
  


TEST_CASE("OmpBLAS gemv", "[OMP]")
{
  
  const int N = 2;
  const int M = 3;
  
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
 
  // Non-batched test
  std::cout << "Testing TRANS gemv" << std::endl;
  test_gemv<float>(M, N, 'T');
  test_gemv<double>(M, N, 'T');
#if defined(QMC_COMPLEX)
  test_gemv<std::complex<float>>(N, M, 'T');
  test_gemv<std::complex<double>>(N, M, 'T');
#endif
 
  /*
  // Batched Test
  std::cout << "Testing TRANS gemv_batched" << std::endl;
    test_gemv_batched<float>(N, N, 'T', 14);
  //  test_gemv_batched<double>(N, N, 'T', 14);
#if defined(QMC_COMPLEX)
  //test_gemv<std::complex<float>>(N, N, 'T', 13);
  //test_gemv<std::complex<double>>(N, N, 'T', 13);
#endif
  */
}
} // namespace qmcplusplus
