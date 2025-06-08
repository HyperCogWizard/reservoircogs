# Generate AtomSpace types from script
OPENCOG_ADD_ATOM_TYPES(reservoir_types.script 
    reservoir_types.h
    reservoir_types.definitions
    reservoir_types.inheritance
    reservoir_types_init.cc
)

# Build types library
ADD_LIBRARY(reservoircogs-types SHARED
    reservoir_types_init.cc
)

TARGET_LINK_LIBRARIES(reservoircogs-types
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install headers
INSTALL(FILES
    reservoir_types.h
    DESTINATION include/opencog/reservoir/types
)