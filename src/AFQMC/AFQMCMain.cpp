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


#include "AFQMC/AFQMCMain.h"
#include "AFQMC/AFQMCFactory.h"

namespace qmcplusplus
{
namespace afqmc
{
// the main function running AFQMC
int AFQMCMain(boost::mpi3::communicator& comm_, xmlNodePtr root, std::ostream& printout)
{
  printout << std::endl
            << "/*************************************************\n"
            << " ********  This is an AFQMC calculation   ********\n"
            << " *************************************************" << std::endl;

  AFQMCFactory afqmc_fac(comm_);
  if (!afqmc_fac.parse(root))
  {
    printout << " Error in AFQMCFactory::parse() ." << std::endl;
    return false;
  }
  return afqmc_fac.execute(root);
}

} // namespace afqmc
} // namespace qmcplusplus
