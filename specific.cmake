CPMAddPackage(
  NAME fmt
  GIT_TAG 7.1.3
  GITHUB_REPOSITORY fmtlib/fmt
  OPTIONS "FMT_INSTALL YES" # create an installable target
)

CPMAddPackage("gh:xtensor-stack/xtl#0.7.4")
if(xtl_ADDED)
  message(STATUS "Found xtl: ${xtl_SOURCE_DIR}")
  include_directories(${xtl_SOURCE_DIR}/include)
endif(xtl_ADDED)

CPMAddPackage("gh:xtensor-stack/xtensor#0.24.3")
if(xtensor_ADDED)
  message(STATUS "Found xtensor: ${xtensor_SOURCE_DIR}")
  include_directories(${xtensor_SOURCE_DIR}/include)
endif(xtensor_ADDED)

CPMAddPackage("gh:ericniebler/range-v3#0.10.0")

set(SPECIFIC_LIBS fmt::fmt range-v3)
# remember to turn off the warnings
