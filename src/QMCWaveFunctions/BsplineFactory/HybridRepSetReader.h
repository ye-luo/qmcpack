//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2023 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//////////////////////////////////////////////////////////////////////////////////////


/** @file
 *
 * derived from BsplineReader
 */

#ifndef QMCPLUSPLUS_HYBRIDREP_READER_H
#define QMCPLUSPLUS_HYBRIDREP_READER_H

#include "BsplineReader.h"
#include "SplineSetReader.h"

#if !defined(QMC_COMPLEX)
#include "HybridRepReal.h"
#endif
#include "HybridRepCplx.h"

namespace qmcplusplus
{
/** General HybridRepSetReader to handle any unitcell
 */
template<typename SA>
class HybridRepSetReader : public BsplineReader
{
  using SplineReader = SplineSetReader<typename SA::SplineBase>;
  using DataType     = typename SA::DataType;
  SplineReader spline_reader_;

public:
  HybridRepSetReader(EinsplineSetBuilder* e);

  std::unique_ptr<SPOSet> create_spline_set(const std::string& my_name,
                                            int spin,
                                            const BandInfoGroup& bandgroup) override;

  /** initialize basic parameters of atomic orbitals */
  void initialize_hybridrep_atomic_centers(SA& bspline) const;

  /** initialize construct atomic orbital radial functions from plane waves */
  inline void create_atomic_centers_Gspace(const Vector<std::complex<double>>& cG,
                                           Communicate& band_group_comm,
                                           const int iorb,
                                           const std::complex<double>& rotate_phase,
                                           SA& bspline) const;

  /** transforming planewave orbitals to 3D B-spline orbitals and 1D B-spline radial orbitals in real space.
   * @param spin orbital dataset spin index
   * @param bandgroup band info
   * @param bspline the spline object being worked on
   */
  void initialize_hybrid_pio_gather(const int spin, const BandInfoGroup& bandgroup, SA& bspline) const;
};

#if defined(QMC_COMPLEX)
extern template class HybridRepSetReader<HybridRepCplx<float>>;
#if !defined(QMC_MIXED_PRECISION)
extern template class HybridRepSetReader<HybridRepCplx<double>>;
#endif
#else
extern template class HybridRepSetReader<HybridRepReal<float>>;
extern template class HybridRepSetReader<HybridRepCplx<float>>;
#if !defined(QMC_MIXED_PRECISION)
extern template class HybridRepSetReader<HybridRepReal<double>>;
extern template class HybridRepSetReader<HybridRepCplx<double>>;
#endif
#endif
} // namespace qmcplusplus
#endif
