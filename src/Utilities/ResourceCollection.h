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

#ifndef QMCPLUSPLUS_RESOURCECOLLECTION_H
#define QMCPLUSPLUS_RESOURCECOLLECTION_H

#include <string>
#include <memory>
#include <cstddef>
#include <vector>
#include <variant>
#include "Resource.h"
#include "type_traits/template_types.hpp"
#include "tests/MemoryResource.h"
#include "tests/TestResource2.h"

namespace qmcplusplus
{

using ResourceWrapper = std::variant<UPtr<MemoryResource>,UPtr<TestResource2>>;

class ResourceCollection
{
public:
  ResourceCollection(const std::string& name);
  const std::string& getName() const { return name_; }
  void printResources();

  size_t addResource(ResourceWrapper&& res);
  ResourceWrapper lendResource(size_t id);
  void takebackResource(size_t i, ResourceWrapper&& res);

private:
  const std::string name_;
  std::vector<ResourceWrapper> collection;
};
} // namespace qmcplusplus
#endif
