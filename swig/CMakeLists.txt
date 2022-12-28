cmake_minimum_required(VERSION 3.0)

# Library versioning (used in Mac Python bindings)x
set(SOVERSION 2)

###################
# Swig Settings
###################
if (RUN_SWIG)
    include(FindSWIG)
    # This module finds an installed SWIG. It sets the following variables:
    # SWIG_FOUND - set to true if SWIG is found
    # SWIG_DIR - the directory where swig is installed
    # SWIG_EXECUTABLE - the path to the swig executable
    # SWIG_VERSION   - the version number of the swig executable

    find_package(SWIG 2.0)
    if (NOT SWIG_FOUND)
        message(STATUS "SWIG 2.0 NOT found")
        return()
    else ()
        if (SWIG_VERSION VERSION_GREATER_EQUAL 4.0)
            set(doxygen_switch "-doxygen")
        endif ()
    endif (NOT SWIG_FOUND)
endif (RUN_SWIG)

###################
# Java Bindings
###################
if (JAVA_BINDINGS)
    if (EXISTS ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp OR RUN_SWIG)

        find_package(JNI)
        if (NOT JNI_INCLUDE_DIRS)
            message(STATUS "Java Native Interface NOT found. Java bindings will NOT be compiled.")
        endif ()

        if (RUN_SWIG)
            find_package(Java)
            if (NOT JAVA_COMPILE)
                message(STATUS "Java compiler NOT found. Java bindings will NOT be generated.")
                set(JAVAC_FAIL ON BOOL)
            endif (NOT JAVA_COMPILE)
        endif (RUN_SWIG)

        if (JNI_INCLUDE_DIRS AND NOT JAVAC_FAIL)
            set(DO_JAVA_BINDINGS ON BOOL)
            message(STATUS "Java bindings will be compiled")
        endif ()

    else (EXISTS ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp OR RUN_SWIG)
        message(STATUS "Warning: Java bindings NOT found. Generate using -DRUN_SWIG=ON.")

    endif (EXISTS ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp OR RUN_SWIG)
endif (JAVA_BINDINGS)

if (DO_JAVA_BINDINGS)
    include_directories(
            ${JNI_INCLUDE_DIRS}
    )
    if (RUN_SWIG)
        add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp
                COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/swig/java/org/mnn
                COMMAND ${SWIG_EXECUTABLE} -java -package org.mnn -c++ ${doxygen_switch} -small -O -templatereduce -naturalvar -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include -I${CMAKE_SOURCE_DIR}/schema/current -I${CMAKE_SOURCE_DIR}/tools/train/source/data  -I${CMAKE_SOURCE_DIR}/tools/train/source/datasets -I${CMAKE_SOURCE_DIR}/tools/train/source/grad -I${CMAKE_SOURCE_DIR}/tools/train/source/models -I${CMAKE_SOURCE_DIR}/tools/train/source/nn -I${CMAKE_SOURCE_DIR}/tools/train/source/optimizer -I${CMAKE_SOURCE_DIR}/tools/train/source/parameters -I${CMAKE_SOURCE_DIR}/tools/train/source/transformer -o ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp -outdir ${CMAKE_BINARY_DIR}/swig/java/org/mnn ${CMAKE_SOURCE_DIR}/swig/mnn-java.i
                COMMAND ${JAVA_COMPILE} ARGS "${CMAKE_BINARY_DIR}/swig/java/org/mnn/*.java"
                COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}/swig/java ${JAVA_ARCHIVE} cf mnn.jar org
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/swig/java/mnn.jar ${CMAKE_SOURCE_DIR}/swig/java
                # COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/swig/java
                #                MAIN_DEPENDENCY mnn-java.i
                )
    endif (RUN_SWIG)

    add_library(bindings_java MODULE ${CMAKE_SOURCE_DIR}/swig/java/mnn-java.cpp)
    target_compile_options(bindings_java PRIVATE -fexceptions)
    target_link_libraries(bindings_java ${JNI_LIBRARIES} ${MNN_DEPS})
    set_target_properties(bindings_java PROPERTIES OUTPUT_NAME mnn_java)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/schema/current)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/data)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/datasets)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/models)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/grad)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/nn)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/optimizer)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/parameters)
    target_include_directories(bindings_java PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/transformer)
    if (APPLE)
        set_target_properties(bindings_java PROPERTIES SUFFIX ".dylib")
    endif (APPLE)

    if ("${JAVA_INSTDIR}" STREQUAL "")
        set(JAVA_INSTDIR lib)
    endif ()

    install(TARGETS bindings_java
            LIBRARY DESTINATION "${JAVA_INSTDIR}"
            COMPONENT bindings_java)
    install(FILES ${CMAKE_SOURCE_DIR}/swig/java/mnn.jar
            DESTINATION "${JAVA_INSTDIR}"
            COMPONENT bindings_java)

endif (DO_JAVA_BINDINGS)


###################
# Python
###################

if (PYTHON_BINDINGS)
    if (EXISTS ${CMAKE_SOURCE_DIR}/swig/python/mnn-python.cpp OR RUN_SWIG)
        # find_package(PythonInterp)
        # if (NOT PYTHONINTERP_FOUND)
        #     message(STATUS "Python interpreter NOT found")
        # endif (NOT PYTHONINTERP_FOUND)

        # if (PYTHONINTERP_FOUND AND NOT Python_ADDITIONAL_VERSIONS)
        #     set(Python_ADDITIONAL_VERSIONS "${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
        # endif (PYTHONINTERP_FOUND AND NOT Python_ADDITIONAL_VERSIONS)

        # find_package(PythonLibs)
        # if (NOT PYTHONLIBS_FOUND)
        #     message(STATUS "Python libraries NOT found")
        # endif (NOT PYTHONLIBS_FOUND)

        # if(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)
        #     set(DO_PYTHON_BINDINGS ON BOOL)
        #     message(STATUS "Python bindings will be compiled")
        # endif(PYTHONLIBS_FOUND AND PYTHONINTERP_FOUND)

        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        if (Python3_FOUND)
            set(DO_PYTHON_BINDINGS ON BOOL)
            message(STATUS "Python bindings will be compiled")
        endif(Python3_FOUND)
    else ()
        message(STATUS "Warning: Python bindings NOT found. Generate using -DRUN_SWIG=ON.")

    endif ()
endif (PYTHON_BINDINGS)

if (DO_PYTHON_BINDINGS)

    include_directories(
            ${PYTHON_INCLUDE_PATH}
    )

    if(RUN_SWIG)
        add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/swig/python/mnn-python.cpp ${CMAKE_SOURCE_DIR}/swig/python/mnn/mnn.py
                COMMAND ${SWIG_EXECUTABLE} -python -c++ ${doxygen_switch} -small -O -templatereduce -naturalvar -I${CMAKE_SOURCE_DIR}/include -I${CMAKE_BINARY_DIR}/include -I${CMAKE_SOURCE_DIR}/schema/current -I${CMAKE_SOURCE_DIR}/tools/train/source/data  -I${CMAKE_SOURCE_DIR}/tools/train/source/datasets -I${CMAKE_SOURCE_DIR}/tools/train/source/grad -I${CMAKE_SOURCE_DIR}/tools/train/source/models -I${CMAKE_SOURCE_DIR}/tools/train/source/nn -I${CMAKE_SOURCE_DIR}/tools/train/source/optimizer -I${CMAKE_SOURCE_DIR}/tools/train/source/parameters -I${CMAKE_SOURCE_DIR}/tools/train/source/transformer -o ${CMAKE_SOURCE_DIR}/swig/python/mnn-python.cpp -outdir ${CMAKE_SOURCE_DIR}/swig/python/mnn ${CMAKE_SOURCE_DIR}/swig/mnn-python.i
                MAIN_DEPENDENCY mnn-python.i
                VERBATIM
                )
    endif(RUN_SWIG)

    configure_file(${CMAKE_SOURCE_DIR}/swig/python/mnn/__init__.py.in
            ${CMAKE_SOURCE_DIR}/swig/python/mnn/__init__.py)
    add_library(bindings_python MODULE ${CMAKE_SOURCE_DIR}/swig/python/mnn-python.cpp)
    target_compile_options(bindings_python PRIVATE -fexceptions -frtti)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/schema/current)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/data)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/datasets)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/models)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/grad)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/nn)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/optimizer)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/parameters)
    target_include_directories(bindings_python PRIVATE ${CMAKE_SOURCE_DIR}/tools/train/source/transformer)

    if(APPLE)
        # Don't link against Python library on Mac
        set_target_properties(bindings_python PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
        target_link_libraries(bindings_python ${MNN_DEPS})
    else()
        # Link against Python library on Linux and Windows
        target_link_libraries(bindings_python ${MNN_DEPS})
    endif()

    if(NOT WIN32)
        set_target_properties(bindings_python PROPERTIES
                OUTPUT_NAME _mnn
                PREFIX ""
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/swig/python/mnn/"
                SUFFIX .so )
        if( "${PYTHON_INSTDIR}" STREQUAL "" )
            execute_process(
                    COMMAND
                    ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix='${CMAKE_INSTALL_PREFIX}'))"
                    OUTPUT_VARIABLE PYTHON_INSTDIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            #workaround for https://bugs.launchpad.net/ubuntu/+source/python3-defaults/+bug/1814653
            if(NOT ${PYTHON_INSTDIR} MATCHES "python[0-9].[0-9]")
                execute_process(
                        COMMAND
                        ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,1,prefix='${CMAKE_INSTALL_PREFIX}'))"
                        OUTPUT_VARIABLE PYTHON_INSTDIR
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                )
                set(PYTHON_INSTDIR "${PYTHON_INSTDIR}/dist-packages")
            endif()
        endif()
        if(NOT BINDINGS_ONLY)
            add_dependencies(bindings_python ${MNN_DEPS})
        endif()

        install(TARGETS bindings_python
                LIBRARY DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/__init__.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/mnn.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/pymnn.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
    else(NOT WIN32)
        set_target_properties(bindings_python PROPERTIES
                OUTPUT_NAME _mnn
                PREFIX ""
                SUFFIX .pyd )
        execute_process(
                COMMAND
                ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(1,0,prefix='${CMAKE_INSTALL_PREFIX}'))"
                OUTPUT_VARIABLE PYTHON_INSTDIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT BINDINGS_ONLY)
            add_dependencies(bindings_python ${MNN_DEPS})
        endif()

        install(TARGETS bindings_python
                LIBRARY DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/__init__.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/mnn.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        install(FILES ${CMAKE_SOURCE_DIR}/swig/python/mnn/pymnn.py
                DESTINATION ${PYTHON_INSTDIR}/mnn
                COMPONENT bindings_python)
        add_custom_command(TARGET bindings_python POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/swig/python/mnn/mnn.py ${CMAKE_BINARY_DIR}/bin/${CMAKE_CFG_INTDIR}
                )
    endif(NOT WIN32)
endif (DO_PYTHON_BINDINGS)
