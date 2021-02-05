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
#include "ResourceCollection.h"

namespace qmcplusplus
{

TEST_CASE("Resource", "[utilities]")
{
  auto mem_res = std::make_unique<MemoryResource>("test_res");
  mem_res->data.resize(5);

  std::unique_ptr<MemoryResource> res_copy = std::make_unique<MemoryResource>(*mem_res);
  REQUIRE(res_copy->data.size() == 5);
}

class WFCResourceConsumer
{
public:
  void createResource(ResourceCollection& collection)
  {
    external_memory_handle = std::make_unique<MemoryResource>("test_res");
    external_memory_handle->data.resize(5);
    resource_index = collection.addResource(std::move(external_memory_handle));
  }

  void acquireResource(ResourceCollection& collection)
  {
    external_memory_handle = std::get<UPtr<MemoryResource>>(collection.lendResource(resource_index));
  }

  void releaseResource(ResourceCollection& collection)
  {
    collection.takebackResource(resource_index, std::move(external_memory_handle));
  }

  MemoryResource* getPtr() const { return external_memory_handle.get(); }

private:
  std::unique_ptr<MemoryResource> external_memory_handle;
  size_t resource_index = -1;
};

TEST_CASE("ResourceCollection", "[utilities]")
{
  ResourceCollection res_collection("abc");
  WFCResourceConsumer wfc;
  REQUIRE(wfc.getPtr() == nullptr);

  wfc.createResource(res_collection);
  REQUIRE(wfc.getPtr() == nullptr);

  res_collection.printResources();
  wfc.acquireResource(res_collection);
  REQUIRE(wfc.getPtr() != nullptr);
  REQUIRE(wfc.getPtr()->data.size() == 5);

  wfc.releaseResource(res_collection);
  REQUIRE(wfc.getPtr() == nullptr);

  ResourceCollection res_Bad("bad");
  auto test_resource_2 = std::make_unique<TestResource2>("test2_res");
  auto bad_res_index = res_Bad.addResource(std::move(test_resource_2));
  CHECK_THROWS(wfc.acquireResource(res_Bad));
}

} // namespace qmcplusplus
