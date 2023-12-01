//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////


/** @file HybridRepCplx.h
 *
 * hold HybridRepCplx
 */
#ifndef QMCPLUSPLUS_HYBRIDREP_CPLX_H
#define QMCPLUSPLUS_HYBRIDREP_CPLX_H

#include "QMCWaveFunctions/BsplineFactory/HybridRepCenterOrbitals.h"
#include "CPU/SIMD/inner_product.hpp"
#include "BsplineSet.h"
#include "SplineC2C.h"
#include "SplineC2R.h"

namespace qmcplusplus
{

template<typename ST, typename TT>
struct SplineC2XType
{
  using value = SplineC2R<ST>;
};

template<typename ST, typename TT>
struct SplineC2XType<ST, std::complex<TT>>
{
  using value = SplineC2C<ST>;
};

/** hybrid representation orbitals combining B-spline orbitals on a grid and atomic centered orbitals.
 * @tparam SplineBase B-spline orbital class.
 *
 * Only works with SplineBase class containing complex splines
 */
template<typename ST>
class HybridRepCplx : public SplineC2XType<ST, SPOSet::ValueType>::value, private HybridRepCenterOrbitals<ST>
{
public:
  using SplineBase       = typename SplineC2XType<ST, SPOSet::ValueType>::value;
  using HYBRIDBASE       = HybridRepCenterOrbitals<ST>;
  using DataType         = typename SplineBase::DataType;
  using PointType        = typename SplineBase::PointType;
  using SingleSplineType = typename SplineBase::SingleSplineType;
  using RealType         = typename SplineBase::RealType;
  // types for evaluation results
  using typename SplineBase::GGGVector;
  using typename SplineBase::GradMatrix;
  using typename SplineBase::GradType;
  using typename SplineBase::GradVector;
  using typename SplineBase::HessVector;
  using typename SplineBase::OffloadMWVGLArray;
  using typename SplineBase::ValueMatrix;
  using typename SplineBase::ValueType;
  using typename SplineBase::ValueVector;

private:
  using typename HYBRIDBASE::Region;

  ValueVector psi_AO, d2psi_AO;
  GradVector dpsi_AO;
  Matrix<ST, aligned_allocator<ST>> multi_myV;
  typename HYBRIDBASE::LocationSmoothingInfo info;

  using SplineBase::myG;
  using SplineBase::myH;
  using SplineBase::myL;
  using SplineBase::myV;

public:
  HybridRepCplx(const std::string& my_name) : SplineBase(my_name) {}

  std::string getClassName() const final { return "Hybrid" + SplineBase::getClassName(); }
  std::string getKeyword() const final { return "Hybrid" + SplineBase::getKeyword(); }
  bool isOMPoffload() const final { return false; }

  std::unique_ptr<SPOSet> makeClone() const override { return std::make_unique<HybridRepCplx>(*this); }

  void bcast_tables(Communicate* comm)
  {
    SplineBase::bcast_tables(comm);
    HYBRIDBASE::bcast_tables(comm);
  }

  void gather_tables(Communicate* comm)
  {
    SplineBase::gather_tables(comm);
    HYBRIDBASE::gather_atomic_tables(comm, SplineBase::offset);
  }

  bool read_splines(hdf_archive& h5f) { return HYBRIDBASE::read_splines(h5f) && SplineBase::read_splines(h5f); }

  bool write_splines(hdf_archive& h5f) { return HYBRIDBASE::write_splines(h5f) && SplineBase::write_splines(h5f); }

  void evaluateValue(const ParticleSet& P, const int iat, ValueVector& psi) override
  {
    HYBRIDBASE::evaluate_v(P, iat, myV, info);
    if (info.region == Region::INTER)
      SplineBase::evaluateValue(P, iat, psi);
    else if (info.region == Region::INSIDE)
      SplineBase::assign_v(P.activeR(iat), myV, psi, 0, myV.size() / 2);
    else
    {
      psi_AO.resize(psi.size());
      SplineBase::assign_v(P.activeR(iat), myV, psi_AO, 0, myV.size() / 2);
      SplineBase::evaluateValue(P, iat, psi);
      HYBRIDBASE::interpolate_buffer_v(psi, psi_AO, info.f);
    }
  }


  void evaluateDetRatios(const VirtualParticleSet& VP,
                         ValueVector& psi,
                         const ValueVector& psiinv,
                         std::vector<ValueType>& ratios) override
  {
    if (VP.isOnSphere())
    {
      // resize scratch space
      psi_AO.resize(psi.size());
      if (multi_myV.rows() < VP.getTotalNum())
        multi_myV.resize(VP.getTotalNum(), myV.size());
      HYBRIDBASE::evaluateValuesC2X(VP, multi_myV, info);
      for (int iat = 0; iat < VP.getTotalNum(); ++iat)
      {
        if (info.region == Region::INTER)
          SplineBase::evaluateValue(VP, iat, psi);
        else if (info.region == Region::INSIDE)
        {
          Vector<ST, aligned_allocator<ST>> myV_one(multi_myV[iat], myV.size());
          SplineBase::assign_v(VP.R[iat], myV_one, psi, 0, myV.size() / 2);
        }
        else
        {
          Vector<ST, aligned_allocator<ST>> myV_one(multi_myV[iat], myV.size());
          SplineBase::assign_v(VP.R[iat], myV_one, psi_AO, 0, myV.size() / 2);
          SplineBase::evaluateValue(VP, iat, psi);
          HYBRIDBASE::interpolate_buffer_v(psi, psi_AO, info.f);
        }
        ratios[iat] = simd::dot(psi.data(), psiinv.data(), psi.size());
      }
    }
    else
    {
      for (int iat = 0; iat < VP.getTotalNum(); ++iat)
      {
        evaluateValue(VP, iat, psi);
        ratios[iat] = simd::dot(psi.data(), psiinv.data(), psi.size());
      }
    }
  }

  void mw_evaluateDetRatios(const RefVectorWithLeader<SPOSet>& spo_list,
                            const RefVectorWithLeader<const VirtualParticleSet>& vp_list,
                            const RefVector<ValueVector>& psi_list,
                            const std::vector<const ValueType*>& invRow_ptr_list,
                            std::vector<std::vector<ValueType>>& ratios_list) const final
  {
    BsplineSet::mw_evaluateDetRatios(spo_list, vp_list, psi_list, invRow_ptr_list, ratios_list);
  }

  void evaluateVGL(const ParticleSet& P, const int iat, ValueVector& psi, GradVector& dpsi, ValueVector& d2psi) override
  {
    HYBRIDBASE::evaluate_vgl(P, iat, myV, myG, myL, info);
    if (info.region == Region::INTER)
      SplineBase::evaluateVGL(P, iat, psi, dpsi, d2psi);
    else if (info.region == Region::INSIDE)
      SplineBase::assign_vgl_from_l(P.activeR(iat), psi, dpsi, d2psi);
    else
    {
      psi_AO.resize(psi.size());
      dpsi_AO.resize(psi.size());
      d2psi_AO.resize(psi.size());
      SplineBase::assign_vgl_from_l(P.activeR(iat), psi_AO, dpsi_AO, d2psi_AO);
      SplineBase::evaluateVGL(P, iat, psi, dpsi, d2psi);
      HYBRIDBASE::interpolate_buffer_vgl(psi, dpsi, d2psi, psi_AO, dpsi_AO, d2psi_AO, info);
    }
  }

  void mw_evaluateVGL(const RefVectorWithLeader<SPOSet>& sa_list,
                      const RefVectorWithLeader<ParticleSet>& P_list,
                      int iat,
                      const RefVector<ValueVector>& psi_v_list,
                      const RefVector<GradVector>& dpsi_v_list,
                      const RefVector<ValueVector>& d2psi_v_list) const final
  {
    BsplineSet::mw_evaluateVGL(sa_list, P_list, iat, psi_v_list, dpsi_v_list, d2psi_v_list);
  }

  void mw_evaluateVGLandDetRatioGrads(const RefVectorWithLeader<SPOSet>& spo_list,
                                      const RefVectorWithLeader<ParticleSet>& P_list,
                                      int iat,
                                      const std::vector<const ValueType*>& invRow_ptr_list,
                                      OffloadMWVGLArray& phi_vgl_v,
                                      std::vector<ValueType>& ratios,
                                      std::vector<GradType>& grads) const final
  {
    BsplineSet::mw_evaluateVGLandDetRatioGrads(spo_list, P_list, iat, invRow_ptr_list, phi_vgl_v, ratios, grads);
  }

  void evaluateVGH(const ParticleSet& P,
                   const int iat,
                   ValueVector& psi,
                   GradVector& dpsi,
                   HessVector& grad_grad_psi) override
  {
    APP_ABORT("HybridRepCplx::evaluate_vgh not implemented!");
    HYBRIDBASE::evaluate_vgh(P, iat, myV, myG, myH, info);
    if (info.region == Region::INTER)
      SplineBase::evaluateVGH(P, iat, psi, dpsi, grad_grad_psi);
    else
      SplineBase::assign_vgh(P.activeR(iat), psi, dpsi, grad_grad_psi, 0, myV.size() / 2);
  }

  void evaluateVGHGH(const ParticleSet& P,
                     const int iat,
                     ValueVector& psi,
                     GradVector& dpsi,
                     HessVector& grad_grad_psi,
                     GGGVector& grad_grad_grad_psi) override
  {
    APP_ABORT("HybridRepCplx::evaluate_vghgh not implemented!");
  }

  void evaluate_notranspose(const ParticleSet& P,
                            int first,
                            int last,
                            ValueMatrix& logdet,
                            GradMatrix& dlogdet,
                            ValueMatrix& d2logdet) final
  {
    // bypass SplineBase::evaluate_notranspose
    BsplineSet::evaluate_notranspose(P, first, last, logdet, dlogdet, d2logdet);
  }

  template<class BSPLINESPO>
  friend class HybridRepSetReader;
  template<class BSPLINESPO>
  friend class SplineSetReader;
  friend struct BsplineReader;
};

extern template class HybridRepCplx<float>;
extern template class HybridRepCplx<double>;
} // namespace qmcplusplus
#endif
