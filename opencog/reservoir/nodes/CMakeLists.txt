# Build reservoir nodes library
ADD_LIBRARY(reservoircogs-nodes SHARED
    ReservoirNode.cc
)

TARGET_LINK_LIBRARIES(reservoircogs-nodes
    reservoircogs-types
    ${ATOMSPACE_LIBRARIES}
    ${COGUTIL_LIBRARIES}
)

# Install headers
INSTALL(FILES
    ReservoirNode.h
    DESTINATION include/opencog/reservoir/nodes
)