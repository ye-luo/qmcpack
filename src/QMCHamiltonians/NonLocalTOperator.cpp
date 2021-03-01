//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Ken Esler, kpesler@gmail.com, University of Illinois at Urbana-Champaign
//                    Jeremy McMinnis, jmcminis@gmail.com, University of Illinois at Urbana-Champaign
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////


/**@file NonLocalTOperator.cpp
 *@brief Definition of NonLocalTOperator
 */
#include "NonLocalTOperator.h"
#include "OhmmsData/ParameterSet.h"
#include "ParticleSet.h"
#include "TrialWaveFunction.h"
#include "NonLocalECPotential.h"

namespace qmcplusplus
{
NonLocalTOperator::NonLocalTOperator(size_t N) : scheme_(Scheme::OFF), Nelec(N), Tau(0.01), Alpha(0.0), Gamma(0.0) {}

/** process options related to TMoves
 * @return Tmove version
 *  Turns out this wants the NodePtr to the entire driver block.
 */
void NonLocalTOperator::put(xmlNodePtr cur)
{
  std::string use_tmove = "no";
  ParameterSet m_param;
  m_param.add(Tau, "timeStep");
  m_param.add(Tau, "timestep");
  m_param.add(Tau, "Tau");
  m_param.add(Tau, "tau");
  m_param.add(Alpha, "alpha");
  m_param.add(Gamma, "gamma");
  m_param.add(use_tmove, "nonlocalmove");
  m_param.add(use_tmove, "nonlocalmoves");
  bool success = m_param.put(cur);
  plusFactor   = Tau * Gamma;
  minusFactor  = -Tau * (1.0 - Alpha * (1.0 + Gamma));
  scheme_      = Scheme::OFF;
  std::ostringstream o;
  if (use_tmove == "no")
  {
    scheme_ = Scheme::OFF;
    o << "  Using Locality Approximation";
  }
  else if (use_tmove == "yes" || use_tmove == "v0")
  {
    scheme_ = Scheme::V0;
    o << "  Using Non-local T-moves v0, M. Casula, PRB 74, 161102(R) (2006)";
  }
  else if (use_tmove == "v1")
  {
    scheme_ = Scheme::V1;
    o << "  Using Non-local T-moves v1, M. Casula et al., JCP 132, 154113 (2010)";
  }
  else if (use_tmove == "v3")
  {
    scheme_ = Scheme::V3;
    o << "  Using Non-local T-moves v3, an approximation to v1";
  }
  else
  {
    APP_ABORT("NonLocalTOperator::put unknown nonlocalmove option " + use_tmove);
  }
#pragma omp master
  app_log() << o.str() << std::endl;
}

void NonLocalTOperator::thingsThatShouldBeInMyConstructor(const std::string& non_local_move_option,
                                                          const double tau,
                                                          const double alpha,
                                                          const double gamma)
{
  Tau         = tau;
  Alpha       = alpha;
  Gamma       = gamma;
  plusFactor  = Tau * Gamma;
  minusFactor = -Tau * (1.0 - Alpha * (1.0 + Gamma));
  scheme_ = Scheme::OFF;
  std::ostringstream o;

  if (non_local_move_option == "no")
  {
    scheme_ = Scheme::OFF;
    o << "  Using Locality Approximation";
  }
  else if (non_local_move_option == "yes" || non_local_move_option == "v0")
  {
    scheme_ = Scheme::V0;
    o << "  Using Non-local T-moves v0, M. Casula, PRB 74, 161102(R) (2006)";
  }
  else if (non_local_move_option == "v1")
  {
    scheme_ = Scheme::V1;
    o << "  Using Non-local T-moves v1, M. Casula et al., JCP 132, 154113 (2010)";
  }
  else if (non_local_move_option == "v3")
  {
    scheme_ = Scheme::V3;
    o << "  Using Non-local T-moves v3, an approximation to v1";
  }
  else
  {
    APP_ABORT("NonLocalTOperator::put unknown nonlocalmove option " + non_local_move_option);
  }
  app_log() << o.str() << std::endl;
}
void NonLocalTOperator::reset() { Txy.clear(); }

const NonLocalData* NonLocalTOperator::selectMove(RealType prob, std::vector<NonLocalData>& txy) const
{
  RealType wgt_t = 1.0;
  for (int i = 0; i < txy.size(); i++)
  {
    if (txy[i].Weight > 0)
    {
      wgt_t += txy[i].Weight *= plusFactor;
    }
    else
    {
      wgt_t += txy[i].Weight *= minusFactor;
    }
  }
  prob *= wgt_t;
  RealType wsum = 1.0;
  int ibar      = 0;
  while (wsum < prob)
  {
    wsum += txy[ibar].Weight;
    ibar++;
  }
  return ibar > 0 ? &(txy[ibar - 1]) : nullptr;
}

void NonLocalTOperator::group_by_elec()
{
  Txy_by_elec.resize(Nelec);
  for (int i = 0; i < Nelec; i++)
  {
    Txy_by_elec[i].clear();
  }

  for (int i = 0; i < Txy.size(); i++)
  {
    Txy_by_elec[Txy[i].PID].push_back(Txy[i]);
  }
}

int NonLocalTOperator::makeNonLocalMoves(ParticleSet& P, TrialWaveFunction& Psi, NonLocalECPotential& nlpp, RandomGenerator_t& myRNG)
{
  using GradType = TrialWaveFunction::GradType;
  NonLocalTOperator& nonLocalOps = nlpp.getNonLocalOps();

  int NonLocalMoveAccepted = 0;
  if (nonLocalOps.getScheme() == NonLocalTOperator::Scheme::V0)
  {
    const NonLocalData* oneTMove = nonLocalOps.selectMove(myRNG());
    //make a non-local move
    if (oneTMove)
    {
      int iat = oneTMove->PID;
      Psi.prepareGroup(P, P.getGroupID(iat));
      if (P.makeMoveAndCheck(iat, oneTMove->Delta))
      {
        GradType grad_iat;
        Psi.calcRatioGrad(P, iat, grad_iat);
        Psi.acceptMove(P, iat, true);
        P.acceptMove(iat);
        NonLocalMoveAccepted++;
      }
    }
  }
  else if (nonLocalOps.getScheme() == NonLocalTOperator::Scheme::V1)
  {
    GradType grad_iat;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      Psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        nlpp.computeOneElectronTxy(P, iat);
        const NonLocalData* oneTMove = nonLocalOps.selectMove(myRNG());
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta))
          {
            Psi.calcRatioGrad(P, iat, grad_iat);
            Psi.acceptMove(P, iat, true);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }
  else if (nonLocalOps.getScheme() == NonLocalTOperator::Scheme::V3)
  {
    nlpp.markAllElecsUnaffected(P);
    nonLocalOps.group_by_elec();
    GradType grad_iat;
    //make a non-local move per particle
    for (int ig = 0; ig < P.groups(); ++ig) //loop over species
    {
      Psi.prepareGroup(P, ig);
      for (int iat = P.first(ig); iat < P.last(ig); ++iat)
      {
        const NonLocalData* oneTMove;
        if (nlpp.isElecAffected(iat))
        {
          // recompute Txy for the given electron effected by Tmoves
          nlpp.computeOneElectronTxy(P, iat);
          oneTMove = nonLocalOps.selectMove(myRNG());
        }
        else
          oneTMove = nonLocalOps.selectMove(myRNG(), iat);
        if (oneTMove)
        {
          if (P.makeMoveAndCheck(iat, oneTMove->Delta))
          {
            Psi.calcRatioGrad(P, iat, grad_iat);
            Psi.acceptMove(P, iat, true);
            // mark all affected electrons
            nlpp.markAffectedElecs(P, iat);
            P.acceptMove(iat);
            NonLocalMoveAccepted++;
          }
        }
      }
    }
  }

  if (NonLocalMoveAccepted > 0)
    Psi.completeUpdates();

  return NonLocalMoveAccepted;
}

} // namespace qmcplusplus
