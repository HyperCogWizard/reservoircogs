CMAKE_MINIMUM_REQUIRED(VERSION 3.16)
PROJECT(reservoircogs)

# Set version
SET(RESERVOIRCOGS_VERSION_MAJOR 0)
SET(RESERVOIRCOGS_VERSION_MINOR 3)
SET(RESERVOIRCOGS_VERSION_PATCH 13)

# OpenCog integration
FIND_PACKAGE(AtomSpace 5.0.3 CONFIG REQUIRED)
FIND_PACKAGE(CogUtil CONFIG REQUIRED)

# C++ Standard
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/opencog
    ${ATOMSPACE_INCLUDE_DIR}
    ${COGUTIL_INCLUDE_DIR}
)

# Build options
OPTION(UNIT_TESTS "Build unit tests" ON)

# Add subdirectories
ADD_SUBDIRECTORY(opencog)

IF(UNIT_TESTS)
    ENABLE_TESTING()
    ADD_SUBDIRECTORY(tests)
ENDIF(UNIT_TESTS)

# Install configuration
INSTALL(TARGETS reservoircogs-types EXPORT ReservoirCogsTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Export targets
INSTALL(EXPORT ReservoirCogsTargets
    FILE ReservoirCogsTargets.cmake
    NAMESPACE ReservoirCogs::
    DESTINATION lib/cmake/ReservoirCogs
)

# Create package config
INCLUDE(CMakePackageConfigHelpers)
WRITE_BASIC_PACKAGE_VERSION_FILE(
    ReservoirCogsConfigVersion.cmake
    VERSION ${RESERVOIRCOGS_VERSION_MAJOR}.${RESERVOIRCOGS_VERSION_MINOR}.${RESERVOIRCOGS_VERSION_PATCH}
    COMPATIBILITY AnyNewerVersion
)

CONFIGURE_PACKAGE_CONFIG_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReservoirCogsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirCogsConfig.cmake
    INSTALL_DESTINATION lib/cmake/ReservoirCogs
)

INSTALL(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirCogsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirCogsConfigVersion.cmake
    DESTINATION lib/cmake/ReservoirCogs
)