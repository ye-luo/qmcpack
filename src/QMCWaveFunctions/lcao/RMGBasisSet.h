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

/** @file RMGBasisSet.h
 */
#ifndef QMCPLUSPLUS_RMG_BASISSET_H
#define QMCPLUSPLUS_RMG_BASISSET_H

namespace qmcplusplus
{
/* A basis set for RMG localized orbitals
   *
   * @tparam T : value type
   */
template<typename T>
struct RMGBasisSet
{
  void checkInVariables(opt_variables_type& active)
  {
  }

  void checkOutVariables(const opt_variables_type& active)
  {
  }

  void resetParameters(const opt_variables_type& active)
  {
  }

 /** evaluate VGL
  */

  template<typename LAT, typename T, typename PosType, typename VGL>
  inline void evaluateVGL(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGL& vgl,PosType Tv)
  {
    // must be implemented. interface arguments needs change
  }

  template<typename LAT, typename T, typename PosType, typename VGH>
  inline void evaluateVGH(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGH& vgh)
  {
  }

  template<typename LAT, typename T, typename PosType, typename VGHGH>
  inline void evaluateVGHGH(const LAT& lattice, const T r, const PosType& dr, const size_t offset, VGHGH& vghgh)
  {
  }

 /** evaluate V
  */
  template<typename LAT, typename T, typename PosType, typename VT>
  inline void evaluateV(const LAT& lattice, const T r, const PosType& dr, VT* restrict psi,PosType Tv)
  {
    // must be implemented. interface arguments needs change
  }
};

} // namespace qmcplusplus
#endif
