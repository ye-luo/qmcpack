//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2018 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_AFQMCMain_H
#define QMCPLUSPLUS_AFQMCMain_H

#include <stack>
#include <ostream>
#include "mpi3/communicator.hpp"
#include "OhmmsData/libxmldefs.h"

namespace qmcplusplus
{
namespace afqmc
{
// the main function running AFQMC
int AFQMCMain(boost::mpi3::communicator&, xmlNodePtr, std::ostream&);
} // namespace afqmc
} // namespace qmcplusplus
#endif
