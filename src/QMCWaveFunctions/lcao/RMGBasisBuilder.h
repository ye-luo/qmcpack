//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2020 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


#ifndef QMCPLUSPLUS_RMGBASIS_BUILDER_H
#define QMCPLUSPLUS_RMGBASIS_BUILDER_H


#include "Message/MPIObjectBase.h"
#include "Utilities/ProgressReportEngine.h"
#include "OhmmsData/AttributeSet.h"
#include "io/hdf_archive.h"

namespace qmcplusplus
{
/** atomic basisset builder
   * @tparam COT, CenteredOrbitalType = SoaAtomicBasisSet<RF,SH>
   *
   * Reimplement AtomiSPOSetBuilder.h
   */
template<typename VALT>
class AOBasisBuilder<RMGBasisSet<VALT>> : public MPIObjectBase
{
private:
  using COT = RMGBasisSet<VALT>;
public:
  AOBasisBuilder(const std::string& eName, Communicate* comm)
      : MPIObjectBase(comm)
  { }

  bool put(xmlNodePtr cur)
  { }
  bool putH5(hdf_archive& hin)
  { }

  COT* createAOSet(xmlNodePtr cur)
  { }
  COT* createAOSetH5(hdf_archive& hin)
  { }
};

} // namespace qmcplusplus
#endif
