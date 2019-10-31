//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2019 QMCPACK developers.
//
// File developed by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Laboratory
//
// File refactored from: MCWalkerConfiguration.cpp, QMCUpdate.cpp
//////////////////////////////////////////////////////////////////////////////////////

#include "QMCDrivers/MCPopulation.h"

#include "Configuration.h"
#include "Concurrency/TasksOneToOne.hpp"
#include "Message/CommOperators.h"

namespace qmcplusplus
{
MCPopulation::MCPopulation(int num_ranks,
                           MCWalkerConfiguration& mcwc,
                           ParticleSet& elecs,
                           TrialWaveFunction& trial_wf,
                           QMCHamiltonian& hamiltonian,
                           int this_rank)
    : num_ranks_(num_ranks), gold_elec_particle_set_(elecs), gold_trial_wavefunction_(trial_wf), gold_hamiltonian_(hamiltonian),
    walker_elec_particle_sets_{elecs}, walker_trial_wavefunctions_{trial_wf}, walker_hamiltonians_{hamiltonian}, rank_(this_rank)
{
  walker_offsets_     = mcwc.WalkerOffsets;
  num_global_walkers_ = mcwc.GlobalNumWalkers;
  num_local_walkers_  = mcwc.LocalNumWalkers;
  num_particles_      = mcwc.getParticleNum();

  // MCWalkerConfiguration doesn't give actual number of groups
  num_groups_ = mcwc.groups();
  particle_group_indexes_.resize(num_groups_);
  for (int i = 0; i < num_groups_; ++i)
  {
    particle_group_indexes_[i].first  = mcwc.first(i);
    particle_group_indexes_[i].second = mcwc.last(i);
  }
  ptclgrp_mass_.resize(num_groups_);
  for (int ig = 0; ig < num_groups_; ++ig)
    ptclgrp_mass_[ig] = mcwc.Mass[ig];
  ptclgrp_inv_mass_.resize(num_groups_);
  for (int ig = 0; ig < num_groups_; ++ig)
    ptclgrp_inv_mass_[ig] = 1.0 / ptclgrp_mass_[ig];
  ptcl_inv_mass_.resize(num_particles_);
  for (int ig = 0; ig < num_groups_; ++ig)
  {
    for (int iat = particle_group_indexes_[ig].first; iat < particle_group_indexes_[ig].second; ++iat)
      ptcl_inv_mass_[iat] = ptclgrp_inv_mass_[ig];
  }
}

/** Default creates walkers equal to num_local_walkers_ and zeroed positions
 */
void MCPopulation::createWalkers() { createWalkers(num_local_walkers_); }

void MCPopulation::createWalkerInplace(UPtr<MCPWalker>& walker_ptr)
{
  //SO this would be where the walker reuse hack would go
  walker_ptr    = std::make_unique<MCPWalker>(num_particles_);
  walker_ptr->R = gold_elec_particle_set_.R;
  walker_ptr->registerData();
  walker_ptr->Properties = gold_elec_particle_set_.Properties;
}

/** Creates walkers with starting positions pos and a clone of the electron particle set and trial wavefunction
 *  
 *  Needed
 *  in: DataSet.size()
 *  in.
 */
void MCPopulation::createWalkers(IndexType num_walkers)
{
  num_local_walkers_ = num_walkers;
  // Ye: need to resize walker_t and ParticleSet Properties
  gold_elec_particle_set_.Properties.resize(1, gold_elec_particle_set_.PropertyList.size());

  walkers_.resize(num_walkers);

  for (auto& walker_ptr : walkers_)
  {
    createWalkerInplace(walker_ptr);
  }

  int num_walkers_created = 0;
  for (auto& walker_ptr : walkers_)
  {
    if (walker_ptr->ID == 0)
    {
      // And so walker ID's start at one because 0 is magic.
      // TODO: This is C++ all indexes start at 0, make uninitialized ID = -1
      walker_ptr->ID       = (++num_walkers_created) * num_ranks_ + rank_;
      walker_ptr->ParentID = walker_ptr->ID;
    }
  }

  outputManager.pause();

  const size_t num_owned_clones = owned_elec_particle_sets_.size();
  // need to make more clones if owned copies + gold is not sufficient
  if (num_owned_clones < num_walkers - 1)
  {
    owned_elec_particle_sets_.resize(num_walkers-1);
    owned_trial_wavefunctions_.resize(num_walkers-1);
    owned_hamiltonians_.resize(num_walkers-1);

    for(size_t i = num_owned_clones; i < num_walkers-1; i++)
    {
      owned_elec_particle_sets_.at(i).reset(new ParticleSet(gold_elec_particle_set_));
      owned_trial_wavefunctions_.at(i).reset(gold_trial_wavefunction_.makeClone(*owned_elec_particle_sets_.at(i)));
      owned_hamiltonians_.at(i).reset(gold_hamiltonian_.makeClone(*owned_elec_particle_sets_.at(i), *owned_trial_wavefunctions_.at(i)));

      walker_elec_particle_sets_.push_back(*owned_elec_particle_sets_.at(i));
      walker_trial_wavefunctions_.push_back(*owned_trial_wavefunctions_.at(i));
      walker_hamiltonians_.push_back(*owned_hamiltonians_.at(i));
    }
  }

  outputManager.resume();

  RefVector<WFBuffer> mcp_wfbuffers;
  mcp_wfbuffers.reserve(num_walkers);
  std::for_each(walkers_.begin(), walkers_.end(),
                [&mcp_wfbuffers](auto& walker) { mcp_wfbuffers.push_back((*walker).DataSet); });

  TrialWaveFunction::flex_registerData(walker_trial_wavefunctions_, walker_elec_particle_sets_, mcp_wfbuffers);

  std::for_each(walkers_.begin(), walkers_.end(), [](auto& walker) { (*walker).DataSet.allocate(); });
}


/** creates a walker and returns a reference
 *
 *  none of the objects are "reused"
 *  if an objects allocation is expensive this should be dealt with
 *  by reusing memory.
 */
MCPopulation::MCPWalker& MCPopulation::spawnWalker()
{
  ++num_local_walkers_;
  outputManager.pause();
  if (dead_walkers_.size() > 0)
  {
    walkers_.push_back(std::move(dead_walkers_.back()));
    dead_walkers_.pop_back();
  }
  else
  {
    walkers_.push_back(std::make_unique<MCPWalker>(num_particles_));
    walkers_.back()->R = gold_elec_particle_set_.R;
    walkers_.back()->registerData();
    walkers_.back()->Properties = gold_elec_particle_set_.Properties;  
  }
  /*
  walker_elec_particle_sets_.push_back(std::make_unique<ParticleSet>(*elec_particle_set_));
  walker_trial_wavefunctions_.push_back(UPtr<TrialWaveFunction>{});
  walker_trial_wavefunctions_.back().reset(trial_wf_->makeClone(*(walker_elec_particle_sets_.back())));
  walker_hamiltonians_.push_back(UPtr<QMCHamiltonian>{});
  walker_hamiltonians_.back().reset(
      hamiltonian_->makeClone(*(walker_elec_particle_sets_.back()), *(walker_trial_wavefunctions_.back())));  
  walker_trial_wavefunctions_.back()->registerData(*(walker_elec_particle_sets_.back()), walkers_.back()->DataSet);
  */
  outputManager.resume();
  return *(walkers_.back());
}

/** Kill last walker
 */
void MCPopulation::killLastWalker()
{
  --num_local_walkers_;
  dead_walkers_.push_back(std::move(walkers_.back()));
  walkers_.pop_back();
  walker_elec_particle_sets_.pop_back();
  walker_trial_wavefunctions_.pop_back();
  walker_hamiltonians_.pop_back();
}
/** Kill a walker
 */
void MCPopulation::killWalker(MCPWalker& walker)
{
  --num_local_walkers_;
  // find the walker and null its pointer in the walker vector
  auto it_walkers = walkers_.begin();
  auto it_psets   = walker_elec_particle_sets_.begin();
  auto it_twfs    = walker_trial_wavefunctions_.begin();
  auto it_hams    = walker_hamiltonians_.begin();
  while (it_walkers != walkers_.end())
  {
    if (&walker == (*it_walkers).get())
    {
      dead_walkers_.push_back(std::move(*it_walkers));
      walkers_.erase(it_walkers);
      walker_elec_particle_sets_.erase(it_psets);
      walker_trial_wavefunctions_.erase(it_twfs);
      walker_hamiltonians_.erase(it_hams);
      return;
    }
    ++it_walkers;
    ++it_psets;
    ++it_twfs;
    ++it_hams;
  }
  throw std::runtime_error("Attempt to kill nonexistent walker in MCPopulation!");
}

QMCTraits::IndexType MCPopulation::update_num_global_walkers(Communicate* comm)
{
  int ncontexts = comm->size();
  std::vector<int> nw(ncontexts, 0);
  std::vector<int> nwoff(ncontexts + 1, 0);

  nw[comm->rank()] = num_local_walkers_;
  comm->allreduce(nw);

  for (int ip = 0; ip < ncontexts; ++ip)
    nwoff[ip + 1] = nwoff[ip] + nw[ip];

  num_global_walkers_ = nwoff[ncontexts];
  walker_offsets_     = nwoff;
  return num_global_walkers_;
}

/** Creates walkers doing their first touch in their crowd (thread) context
 *
 *  This is basically premature optimization but I wanted to check if this sort of thing
 *  would work.  It seems to. This sort of structure not an #omp parallel section must be used in the driver.
 *  No new bare openmp directives should be added to code with updated design.
 */
// void MCPopulation::createWalkers(int num_crowds,
//                                  int walkers_per_crowd,
//                                  IndexType num_walkers,
//                                  const ParticleAttrib<TinyVector<QMCTraits::RealType, 3>>& positions)
// {
//   walkers_.resize(num_walkers);

//   TasksOneToOne<> do_per_crowd(num_crowds);

//   std::vector<std::unique_ptr<std::vector<std::unique_ptr<MCPWalker>>>> walkers_per_crowd_per_slot;
//   walkers_per_crowd_per_slot.resize(num_crowds);
//   auto first_touch_create_walkers =
//       [this, &walkers_per_crowd,
//        &positions](int crowd_index, std::vector<std::unique_ptr<std::vector<std::unique_ptr<MCPWalker>>>>& wpcps) {
//         wpcps[crowd_index] = std::make_unique<std::vector<std::unique_ptr<MCPWalker>>>(walkers_per_crowd);
//         std::vector<std::unique_ptr<MCPWalker>>& this_crowds_walkers = *(wpcps[crowd_index]);
//         this_crowds_walkers.resize(walkers_per_crowd);
//         for (int i = 0; i < walkers_per_crowd; ++i)
//         {
//           std::unique_ptr<MCPWalker>& walker_uptr = this_crowds_walkers[i];
//           walker_uptr.reset(new MCPWalker(num_particles_));
//           walker_uptr->R.resize(num_particles_);
//           walker_uptr->R = positions;
//         }
//       };
//   do_per_crowd(first_touch_create_walkers, walkers_per_crowd_per_slot);

//   auto walkers_it = walkers_.begin();
//   std::for_each(walkers_per_crowd_per_slot.begin(), walkers_per_crowd_per_slot.end(),
//                 [&walkers_it](std::unique_ptr<std::vector<std::unique_ptr<MCPWalker>>>& per_crowd_ptr) {
//                   std::vector<std::unique_ptr<MCPWalker>>& walkers_per_crowd = *per_crowd_ptr;
//                   for (int i = 0; i < walkers_per_crowd.size(); ++i)
//                   {
//                     *walkers_it = std::move(walkers_per_crowd[i]);
//                     ++walkers_it;
//                   }
//                 });
// }


} // namespace qmcplusplus
