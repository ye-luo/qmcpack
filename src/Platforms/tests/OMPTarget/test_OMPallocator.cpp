//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#include "catch.hpp"

#include <memory>
#include <iostream>
#include "OMPTarget/OMPallocator.hpp"
#include "OhmmsPETE/OhmmsVector.h"

namespace qmcplusplus
{
TEST_CASE("OMPTarget_allocators", "[OMPTarget]")
{
  constexpr size_t N = 1024;
  { // OMPallocator, dual space
    Vector<double, OMPallocator<double>> vec(N);
    auto* ptr = vec.data();
    PRAGMA_OFFLOAD("omp target teams distribute parallel for")
    for (int i = 0; i < N; i++)
      ptr[i] = i;
  }
  { // OMPDeviceAllocator, device space
    Vector<double, OMPDeviceAllocator<double>> vec(N);
    auto* ptr = vec.data();
    PRAGMA_OFFLOAD("omp target teams distribute parallel for is_device_ptr(ptr)")
    for (int i = 0; i < N; i++)
      ptr[i] = i;
  }
}

} // namespace qmcplusplus
