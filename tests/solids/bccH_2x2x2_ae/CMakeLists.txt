if(NOT QMC_CUDA)
  if(NOT MIXED_PRECISION)
  simple_run_and_check(short-bccH_2x2x2_ae-grad_lap "${qmcpack_SOURCE_DIR}/tests/solids/bccH_2x2x2_ae" grad_lap.xml 1 1
                       check_grad_lap.py)

  simple_run_and_check(short-bccH_2x2x2_ae-deriv "${qmcpack_SOURCE_DIR}/tests/solids/bccH_2x2x2_ae" deriv.xml 1 1
                       check_deriv.py)

  simple_run_and_check(short-bccH_2x2x2_ae-gamma-deriv "${qmcpack_SOURCE_DIR}/tests/solids/bccH_2x2x2_ae"
                       gamma_deriv.xml 1 1 check_deriv.py)
  else()
    message_verbose("Skipping bccH_2x2x2_ae derivative tests in mixed precision (QMC_MIXED_PRECISION=1)")
  endif(NOT MIXED_PRECISION)
else()
  message_verbose("Skipping bccH_2x2x2_ae derivative tests because they are not support by CUDA build (QMC_CUDA=1)")
endif()
