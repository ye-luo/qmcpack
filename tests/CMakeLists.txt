message(STATUS "Adding integration tests for QMCPACK")
if(NOT QMC_VERBOSE_CONFIGURATION)
  message("Set QMC_VERBOSE_CONFIGURATION=1 for details of which tests will be enabled.")
endif()

add_subdirectory(converter)
add_subdirectory(estimator)
add_subdirectory(models)
add_subdirectory(io)
add_subdirectory(performance)

if(QE_FOUND)
  add_subdirectory(pw2qmcpack)
endif()

message("Adding system tests for QMCPACK")

if(NOT TEST_MAX_PROCS)
  set(TEST_MAX_PROCS 16)
  message("TEST_MAX_PROCS was unset. Set to 16")
endif()

set(QMCAPP_ERR "")
if((NOT ${TEST_MAX_PROCS}) OR (${TEST_MAX_PROCS} STREQUAL ""))
  set(QMCAPP_ERR "${QMCAPP_ERR}  TEST_MAX_PROCS not set: '${TEST_MAX_PROCS}'\n")
endif()

if(MIXED_PRECISION)
  set(SUCCESS_STATUS_MP FALSE)
else()
  set(SUCCESS_STATUS_MP TRUE)
endif()

if(ENABLE_OFFLOAD)
  set(SUCCESS_STATUS_OFFLOAD FALSE)
else()
  set(SUCCESS_STATUS_OFFLOAD TRUE)
endif()

if(QMC_CUDA)
  set(SUCCESS_STATUS_CUDA FALSE)
else()
  set(SUCCESS_STATUS_CUDA TRUE)
endif()

if(BUILD_AFQMC)
  include("${qmcpack_SOURCE_DIR}/CMake/python.cmake")
  set(afqmctools_python_reqs numpy;h5py;mpi4py;scipy;pyscf)
  check_python_reqs(afqmctools_python_reqs afqmc_workflow add_tests)
  if(add_tests)
    message("Adding tests for AFQMC workflow.")
    add_subdirectory("afqmc/workflow")
  endif()
endif()

if(HAVE_PYSCF)
  if(NOT QMC_COMPLEX)
    include("${qmcpack_SOURCE_DIR}/CMake/python.cmake")
    set(pyscf_python_reqs numpy;h5py;pyscf;PyscfToQmcpack;PyscfToQmcpack_Spline)
    check_python_reqs(pyscf_python_reqs pyscf_workflow add_tests)

    if(add_tests)
      message("Python dependencies met. Adding PySCF workflow tests")
      add_subdirectory("pyscf")
    endif()
  endif()
endif()
#
#
if(NOT ${QMCAPP_ERR} STREQUAL "")
  message("${QMCAPP_ERR}  skipping tests")
else()
  add_subdirectory("heg/heg_14_gamma")
  add_subdirectory("heg/heg_54_J2rpa")
  add_subdirectory("molecules")
  add_subdirectory("solids")
endif()
