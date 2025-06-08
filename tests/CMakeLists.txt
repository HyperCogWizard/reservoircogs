# Test configuration
IF(UNIT_TESTS)
    FIND_PACKAGE(GTest REQUIRED)
    
    # Test executable
    ADD_EXECUTABLE(test_reservoir_node test_reservoir_node.cc)
    
    TARGET_LINK_LIBRARIES(test_reservoir_node
        reservoircogs-nodes
        reservoircogs-algorithms
        ${GTEST_LIBRARIES}
        ${GTEST_MAIN_LIBRARIES}
        ${ATOMSPACE_LIBRARIES}
        ${COGUTIL_LIBRARIES}
        pthread
    )
    
    # Add to test suite
    ADD_TEST(NAME reservoir_node_tests COMMAND test_reservoir_node)
ENDIF(UNIT_TESTS)