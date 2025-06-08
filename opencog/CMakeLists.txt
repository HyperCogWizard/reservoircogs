ADD_SUBDIRECTORY(reservoir)

# Build options
OPTION(BUILD_EXAMPLES "Build example programs" ON)

# Add examples
IF(BUILD_EXAMPLES)
    ADD_SUBDIRECTORY(../examples/atomspace examples)
ENDIF(BUILD_EXAMPLES)