# Build reservoir algorithms library
ADD_LIBRARY(reservoircogs-algorithms SHARED
    ReservoirAlgorithms.cc
)

TARGET_LINK_LIBRARIES(reservoircogs-algorithms
    reservoircogs-nodes
    reservoircogs-types
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install headers
INSTALL(FILES
    ReservoirAlgorithms.h
    PSystems.h
    DESTINATION include/opencog/reservoir/algorithms
)