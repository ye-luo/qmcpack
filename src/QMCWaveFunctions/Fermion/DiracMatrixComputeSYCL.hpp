//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021 QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@intel.com, Intel corporation
//
// File created by: Peter Doak, doakpw@ornl.gov, Oak Ridge National Lab
//////////////////////////////////////////////////////////////////////////////////////

#ifndef QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SYCL_MKL_H
#define QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SYCL_MKL_H

#include "OhmmsPETE/OhmmsMatrix.h"
#include "OMPTarget/OMPallocator.hpp"
#include "Platforms/DualAllocatorAliases.hpp"
#include "Concurrency/OpenMP.h"
#include "CPU/SIMD/simd.hpp"
#include "SYCL/syclBLAS.hpp"
#include "SYCL/syclSolver.hpp"
#include "QMCWaveFunctions/detail/SYCL/sycl_determinant_helper.hpp"
#include "ResourceCollection.h"

//#define MKL_BATCHED_INVERSE

namespace qmcplusplus
{

/** class to compute matrix inversion and the log value of determinant
 *  of a batch of DiracMatrixes.
 *
 *  @tparam VALUE_FP the datatype used in the actual computation of the matrix
 *  
 *  There is one per crowd not one per MatrixUpdateEngine.
 *  this puts ownership of the scratch resources in a sensible place.
 *
 *  This is compatible with DiracMatrixComputeOMPTarget and can be used both on CPU
 *  and GPU when the resrouce management is properly handled.
 */
template<typename VALUE_FP>
class DiracMatrixComputeSYCL : public Resource
{
public:
  using FullPrecReal = RealAlias<VALUE_FP>;
  using LogValue     = std::complex<FullPrecReal>;

  template<typename T>
  using DualMatrix = Matrix<T, PinnedDualAllocator<T>>;

  template<typename T>
  using DualVector = Vector<T, PinnedDualAllocator<T>>;

  //sycl::queue managed by MatrixDelayedUpdateSYCL
  using HandleResource = sycl::queue;

private:

  template<typename T>
  using DeviceVector = Vector<T, SYCLAllocator<T>>;

  unsigned norb_         = 0;
  unsigned lda_          = 0;
  unsigned batch_size_   = 0;
  unsigned getrf_ws      = 0;
  unsigned getri_ws      = 0;
  unsigned lwork_        = 0;

  DeviceVector<VALUE_FP>     psiM_fp_;
  DeviceVector<VALUE_FP>     m_work_;  
  DeviceVector<std::int64_t> pivots_;

  /** reset internal work space.
   */
  inline void reset(HandleResource& resource,  const int n, const int lda, const int batch_size)
  {

    norb_    = n;
    lda_     = lda;

    if(batch_size > batch_size_)
    {
      batch_size_ = batch_size;
      psiM_fp_.resize(batch_size*n*lda_);
      pivots_.resize(batch_size*lda_);

      getrf_ws = syclSolver::getrf_scratchpad_size<VALUE_FP>(resource,n,n,lda);
      getri_ws = syclSolver::getri_scratchpad_size<VALUE_FP>(resource,n,lda);
      lwork_   = std::max(getrf_ws,getri_ws);
      //  getrf_ws = syclSolver::getrf_batch_scratchpad_size<VALUE_FP>(resource, n, n, lda, n*lda, lda, batch_size);
      //  getri_ws = syclSolver::getri_batch_scratchpad_size<VALUE_FP>(resource, n, lda, n*lda, lda, batch_size);
      m_work_.resize(batch_size*lwork_);
    }

  }

  //double in and double out
  inline void invert_single(HandleResource& resource, 
                            VALUE_FP* restrict ainv_gpu,
                            LogValue& log_value,
                            int iw)
  {
    syclSolver::getrf(resource, norb_, norb_, ainv_gpu, lda_, pivots_.data()+lda_*iw,
                      m_work_.data() + lwork_*iw, getrf_ws).wait();
    log_value = computeLogDet_sycl<VALUE_FP>(resource, norb_, lda_, ainv_gpu, pivots_.data()+lda_*iw);
    syclSolver::getri(resource, norb_, ainv_gpu, lda_, pivots_.data()+lda_*iw, m_work_.data()+ lwork_*iw, getri_ws).wait();
  }

public:
  DiracMatrixComputeSYCL() : Resource("DiracMatrixComputeOMPTarget") {}

  Resource* makeClone() const override { return new DiracMatrixComputeSYCL(*this); }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when VALUE_FP and TMAT are the same
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   * \param [in]    resource          compute resource
   * \param [in]    a_mat             matrix to be inverted
   * \param [out]   inv_a_mat         the inverted matrix
   * \param [out]   log_value         breaks compatibility of MatrixUpdateOmpTarget with
   *                                  DiracMatrixComputeCUDA but is fine for OMPTarget        
   */
  template<typename TMAT>
  inline std::enable_if_t<std::is_same<VALUE_FP, TMAT>::value> invert_transpose(HandleResource& resource,
                                                                                DualMatrix<TMAT>& a_mat,
                                                                                DualMatrix<TMAT>& inv_a_mat,
                                                                                LogValue& log_value)
  {
    const size_t n   = a_mat.rows();
    const size_t lda = a_mat.cols();
    const size_t ldb = inv_a_mat.cols();

    reset(resource,n,lda,1);

    syclBLAS::transpose(resource,a_mat.device_data(),n,lda,inv_a_mat.device_data(),n,ldb).wait();
    invert_single(resource, inv_a_mat.device_data(), log_value, 0);
    inv_a_mat.updateFrom();
  }

  /** compute the inverse of the transpose of matrix A and its determinant value in log
   * when VALUE_FP and TMAT are the different
   * @tparam TMAT matrix value type
   * @tparam TREAL real type
   */
  template<typename TMAT>
  inline std::enable_if_t<!std::is_same<VALUE_FP, TMAT>::value> invert_transpose(HandleResource& resource,
                                                                                 DualMatrix<TMAT>& a_mat,
                                                                                 DualMatrix<TMAT>& inv_a_mat,
                                                                                 LogValue& log_value)
  {
    const int n   = a_mat.rows();
    const int lda = a_mat.cols();
    const int ldb = inv_a_mat.cols();

    reset(resource,n,lda,1);

    syclBLAS::transpose(resource,a_mat.device_data(),n,lda,psiM_fp_.data(),n,ldb).wait();
    invert_single(resource, psiM_fp_.data(), log_value, 0);
    syclBLAS::copy_n(resource, psiM_fp_.data(), n*lda, inv_a_mat.device_data()).wait();
    inv_a_mat.updateFrom();
  }

  /** This covers both mixed and Full precision case.
   *  
   *  \todo measure if using the a_mats without a copy to contiguous vector is better.
   */
  template<typename TMAT>
  inline void mw_invertTranspose(HandleResource& resource,
                                 const RefVector<const DualMatrix<TMAT>>& a_mats,
                                 RefVector<DualMatrix<TMAT>>& inv_a_mats,
                                 DualVector<LogValue>& log_values)
  {
    const int nw  = a_mats.size();
    const int n   = a_mats[0].get().rows();
    const int lda = a_mats[0].get().cols();
    const int nsqr{n * lda};

    reset(resource,n,lda,nw);

#ifdef USE_LOOP_FOR_BATCH
#pragma omp parallel for
    for (int iw = 0; iw < nw; ++iw)
    {
      VALUE_FP* psiM_gpu =  psiM_fp_.data() + iw*nsqr;
      syclBLAS::transpose(resource, a_mats[iw].get().device_data(), n, lda, psiM_gpu , n, lda).wait(); 
      invert_single(resource, psiM_gpu, log_values[iw], iw);
      syclBLAS::copy_n(resource, psiM_gpu, n*lda, inv_a_mats[iw].get().device_data()).wait();
      inv_a_mats[iw].get().updateFrom();
    }
#else
    std::vector<sycl::event> batch_events(nw);
    //THIS needs to be fully batched
    //type conversion and transpose to a strided psiM_fp_ in double
    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw] = syclBLAS::transpose(resource, a_mats[iw].get().device_data(), n, lda, psiM_fp_.data()+iw*nsqr,n,lda); 
    sycl::event::wait(batch_events);

#ifdef MKL_BATCHED_INVERSE
    try
    {
      syclSolver::getrf_batch(resource,n,n,psiM_fp_.data(),lda, nsqr,
                              pivots_.data(), lda, batch_size_, m_work_.data(), getrf_ws*nw).wait();
    }
    catch (sycl::exception const& ex)
    {
      std::ostringstream err;
      err << "\t\tCaught SYCL exception during getrf:\n" << ex.what() << "  status: " << ex.code() << std::endl;
      std::cerr << err.str();
      throw std::runtime_error(err.str());
    }
#else
    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw]= syclSolver::getrf(resource, n, n, psiM_fp_.data()+iw*nsqr, lda, 
                                          pivots_.data()+iw*lda, m_work_.data()+iw*lwork_, getrf_ws);
    sycl::event::wait(batch_events);
#endif

    //THIS needs to be fully batched
    for (int iw = 0; iw < nw; ++iw)
      log_values[iw] = computeLogDet_sycl<VALUE_FP>(resource, norb_, lda_, psiM_fp_.data() + iw*nsqr, pivots_.data() + iw*lda);

#ifdef MKL_BATCHED_INVERSE
    syclSolver::getri_batch(resource,n,psiM_fp_.data(), lda, nsqr,
                            pivots_.data(), lda, batch_size_, m_work_.data(),getri_ws*nw).wait();
#else
    for (int iw = 0; iw < nw; ++iw)
      batch_events[iw] = syclSolver::getri(resource, n, psiM_fp_.data()+iw*nsqr,lda, 
                                           pivots_.data()+lda*iw, m_work_.data()+iw*lwork_, getri_ws);
    sycl::event::wait(batch_events);
#endif

    //THIS needs to be fully batched
    for (int iw = 0; iw < nw; ++iw)
    {
      syclBLAS::copy_n(resource, psiM_fp_.data() + nsqr * iw, n*lda, inv_a_mats[iw].get().device_data()).wait();
      inv_a_mats[iw].get().updateFrom(); 
    }
#endif
  }
};
} // namespace qmcplusplus

#endif // QMCPLUSPLUS_DIRAC_MATRIX_COMPUTE_SCYL_H
