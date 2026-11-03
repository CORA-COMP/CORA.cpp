# install - the installable package: headers, the library and the CMake config of find_package(CORA)
# Included by CMakeLists.txt after the library; the variables are its.
#
#   cmake --install build --prefix /opt/cora
#   find_package(CORA REQUIRED)   target_link_libraries(app PRIVATE CORA::cora)    (in a project)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/cora" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.def")
install(FILES "${CMAKE_BINARY_DIR}/include/cora/config.h" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cora")
install(TARGETS cora cora_flags EXPORT CORATargets
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(EXPORT CORATargets NAMESPACE CORA:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CORA")

# An Eigen fetched at configure time has no package of its own: its headers go with CORA.
set(CORA_EIGEN_BUNDLED OFF)
if(DEFINED eigen3_SOURCE_DIR)
  set(CORA_EIGEN_BUNDLED ON)
  install(DIRECTORY "${eigen3_SOURCE_DIR}/Eigen" "${eigen3_SOURCE_DIR}/unsupported"
          DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cora/eigen3")
endif()
set(CORA_HAS_OPENMP OFF)
if(OpenMP_CXX_FOUND AND NOT MINGW)
  set(CORA_HAS_OPENMP ON)
endif()

# the usage requirements of cora_torch, written into CORAConfig.cmake (see CORA::torch there)
if(CORACPP_HAS_TORCH)
  get_target_property(CORA_TORCH_DEFINITIONS cora_torch INTERFACE_COMPILE_DEFINITIONS)
  get_target_property(CORA_TORCH_INCLUDES cora_torch INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(CORA_TORCH_LIBS cora_torch INTERFACE_LINK_LIBRARIES)
endif()

configure_package_config_file("${CMAKE_SOURCE_DIR}/scripts/cmake/CORAConfig.cmake.in"
                              "${CMAKE_BINARY_DIR}/CORAConfig.cmake"
                              INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CORA")
write_basic_package_version_file("${CMAKE_BINARY_DIR}/CORAConfigVersion.cmake" VERSION 0.1.0
                                 COMPATIBILITY SameMinorVersion)
install(FILES "${CMAKE_BINARY_DIR}/CORAConfig.cmake" "${CMAKE_BINARY_DIR}/CORAConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/CORA")

# The test: install into the build, then build and run a tiny program against the install tree.
add_test(NAME install/consumer
         COMMAND "${CMAKE_COMMAND}" "-DBUILD=${CMAKE_BINARY_DIR}" "-DSOURCE=${CMAKE_SOURCE_DIR}"
                 "-DGENERATOR=${CMAKE_GENERATOR}" "-DCXX=${CMAKE_CXX_COMPILER}"
                 "-DCONFIG=$<CONFIG>" -P "${CMAKE_SOURCE_DIR}/scripts/cmake/install_test.cmake")
set_tests_properties(install/consumer PROPERTIES LABELS test)
