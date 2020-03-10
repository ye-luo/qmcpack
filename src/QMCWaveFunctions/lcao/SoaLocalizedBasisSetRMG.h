//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////


/** @file SoaLocalizedBasisSetRMG.h
 * @brief specialization for RMGBasisSet
 *
 */
#ifndef QMCPLUSPLUS_SOA_LOCALIZEDBASISSET_RMG_H
#define QMCPLUSPLUS_SOA_LOCALIZEDBASISSET_RMG_H

namespace qmcplusplus
{
/** specialization for SoaLocalizedBasisSet<RMGBasisSet<T>, ORBT>
 */
template<typename T, typename ORBT>
struct SoaLocalizedBasisSet<RMGBasisSet<T>, ORBT> : public SoaBasisSetBase<ORBT>
{
  using COT = RMGBasisSet<ORBT>;
  using BaseType = SoaBasisSetBase<ORBT>;
  typedef typename BaseType::vgl_type vgl_type;
  typedef typename BaseType::vgh_type vgh_type;
  typedef typename BaseType::vghgh_type vghgh_type;
  typedef typename ParticleSet::PosType PosType;

  using BaseType::BasisSetSize;

  ///number of centers, e.g., ions
  size_t NumCenters;
  ///number of quantum particles
  size_t NumTargets;
  ///ion particle set
  const ParticleSet& ions_;
  ///number of quantum particles
  const int myTableIndex;
  ///Global Coordinate of Supertwist read from HDF5
  PosType SuperTwist;

  /** container to store the offsets of the basis functions
   *
   * the number of basis states for center J is BasisOffset[J+1]-Basis[J]
   */
  aligned_vector<size_t> BasisOffset;

  /** container of the unique pointers to the Atomic Orbitals
   *
   * size of LOBasisSet = number  of unique centers
   */
  aligned_vector<*> LOBasisSet;

  /** constructor
   * @param ions ionic system
   * @param els electronic system
   */
  SoaLocalizedBasisSet(ParticleSet& ions, ParticleSet& els)
      : ions_(ions), myTableIndex(els.addTable(ions, DT_SOA)), SuperTwist(0.0)
  {
    NumCenters = ions.getTotalNum();
    NumTargets = els.getTotalNum();
    LOBasisSet.resize(ions.getSpeciesSet().getTotalNum(), 0);
    BasisOffset.resize(NumCenters + 1);
    BasisSetSize = 0;
  }

  /** copy constructor */
  SoaLocalizedBasisSet(const SoaLocalizedBasisSet& a) = default;

  /** makeClone */
  //SoaLocalizedBasisSet<COT>* makeClone() const
  BaseType* makeClone() const
  {
    SoaLocalizedBasisSet<COT, ORBT>* myclone = new SoaLocalizedBasisSet<COT, ORBT>(*this);
    for (int i = 0; i < LOBasisSet.size(); ++i)
      myclone->LOBasisSet[i] = LOBasisSet[i]->makeClone();
    return myclone;
  }

  void setBasisSetSize(int nbs)
  {
    const auto& IonID(ions_.GroupID);
    if (BasisSetSize > 0 && nbs == BasisSetSize)
      return;

    //evaluate the total basis dimension and offset for each center
    BasisOffset[0] = 0;
    for (int c = 0; c < NumCenters; c++)
    {
      BasisOffset[c + 1] = BasisOffset[c] + LOBasisSet[IonID[c]]->getBasisSetSize();
    }
    BasisSetSize = BasisOffset[NumCenters];
  }

  /** compute VGL 
   * @param P quantum particleset
   * @param iat active particle
   * @param vgl Matrix(5,BasisSetSize)
   * @param trialMove if true, use getTempDists()/getTempDispls()
   */
  inline void evaluateVGL(const ParticleSet& P, int iat, vgl_type& vgl)
  {
    // Ye: needs implemenation
  }


  /** compute VGH 
   * @param P quantum particleset
   * @param iat active particle
   * @param vgl Matrix(10,BasisSetSize)
   * @param trialMove if true, use getTempDists()/getTempDispls()
   */
  inline void evaluateVGH(const ParticleSet& P, int iat, vgh_type& vgh)
  {
  }

  /** compute VGHGH 
   * @param P quantum particleset
   * @param iat active particle
   * @param vghgh Matrix(20,BasisSetSize)
   * @param trialMove if true, use getTempDists()/getTempDispls()
   */
  inline void evaluateVGHGH(const ParticleSet& P, int iat, vghgh_type& vghgh)
  {
  }

  /** compute values for the iat-paricle move
   *
   * Always uses getTempDists() and getTempDispls()
   * Tv is a translation vector; In PBC, in order to reduce the number
   * of images that need to be summed over when generating the AO the 
   * nearest image displacement, dr, is used. Tv corresponds to the 
   * translation that takes the 'general displacement' (displacement
   * between ion position and electron position) to the nearest image 
   * displacement. We need to keep track of Tv because it must be add
   * as a phase factor, i.e., exp(i*k*Tv).
   */
  inline void evaluateV(const ParticleSet& P, int iat, ORBT* restrict vals)
  {
    // Ye: needs implemenation
  }

  inline void evaluateGradSourceV(const ParticleSet& P, int iat, const ParticleSet& ions, int jion, vgl_type& vgl)
  {
  }

  inline void evaluateGradSourceVGL(const ParticleSet& P, int iat, const ParticleSet& ions, int jion, vghgh_type& vghgh)
  {
  }

  /** add a new set of Centered Atomic Orbitals
   * @param icenter the index of the center
   * @param aos a set of Centered Atomic Orbitals
   */
  void add(int icenter, COT* aos) { LOBasisSet[icenter] = aos; }
};
} // namespace qmcplusplus
#endif
