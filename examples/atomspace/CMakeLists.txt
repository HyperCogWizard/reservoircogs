# Build examples
IF(BUILD_EXAMPLES)
    # Basic ESN AtomSpace example
    ADD_EXECUTABLE(basic_esn_example basic_esn_atomspace.cpp)
    TARGET_LINK_LIBRARIES(basic_esn_example
        reservoircogs-nodes
        reservoircogs-algorithms
        ${ATOMSPACE_LIBRARIES}
        ${COGUTIL_LIBRARIES}
    )
    
    # Emotion AtomSpace example
    ADD_EXECUTABLE(emotion_example emotion_atomspace_example.cpp)
    TARGET_LINK_LIBRARIES(emotion_example
        reservoircogs-nodes
        reservoircogs-algorithms
        ${ATOMSPACE_LIBRARIES}
        ${COGUTIL_LIBRARIES}
    )
    
    # Install examples
    INSTALL(TARGETS basic_esn_example emotion_example
        RUNTIME DESTINATION bin/examples
    )
    
    # Python examples (just copy files)
    INSTALL(FILES
        python_cpp_bridge.py
        README.md
        emotion_atomspace_example.cpp
        DESTINATION share/reservoircogs/examples/atomspace
    )
ENDIF(BUILD_EXAMPLES)