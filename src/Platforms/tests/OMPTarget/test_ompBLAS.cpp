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
#include <checkMatrix.hpp>

namespace qmcplusplus
{
TEST_CASE("OmpBLAS gemv", "[OMP]")
{
  int N = 100;
  using vec_t = Vector<double, OMPallocator<double>>;
  using mat_t = Matrix<double, OMPallocator<double>>;

  ompBLAS::ompBLAS_handle handle;
  
  for(int i = 0; i < N; i++)
    {
      handle = i;
      vec_t A(i); // Input vector
      mat_t B(i, i); // Input matrix
      vec_t C(i); // Result vector (test)
      vec_t D(i); // Result vector BLAS
      
      ompBLAS:: gemv(handle, 'T', i, i, 1.0, B.data(), 1, A.data(), 1, 1.0, C.data(), 1); // tests omp gemv
      BLAS:: gemv(i, i, B.data(), A.data(), D.data());

      bool are_same = true;
      int index = 0;
      do
	{
	  are_same = C.operator[](i) == D.operator[](i);
	  index++;
	} while(are_same == true && index < N);

      REQUIRE(are_same == true);
      
    }
}

} // namespace qmcplusplus
