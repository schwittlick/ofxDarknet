cmake_minimum_required(VERSION 2.8)
cmake_policy( SET CMP0042 NEW)
project( darknet C CXX )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

## Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--export-dynamic -Wall -Wno-sign-compare -fPIC")
endif()

#thread
FIND_PACKAGE(Threads)
list(APPEND LIBRARIES ${CMAKE_THREAD_LIBS_INIT})

# CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
    message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O2;-Xcompiler;-fPIC;")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    add_definitions(-DGPU)
    cuda_include_directories(src)
else()
    list(APPEND LIBRARIES "m")
endif()


# ------- INCLUDES ---------
set( SRC_DIR ${PROJECT_SOURCE_DIR}/../include )
include_directories( ${SRC_DIR} )

FILE(GLOB SRC_FILES         ${SRC_DIR}/*.c)
LIST(REMOVE_ITEM SRC_FILES "${SRC_DIR}/server.c")
LIST(REMOVE_ITEM SRC_FILES "${SRC_DIR}/old.c")
LIST(REMOVE_ITEM SRC_FILES "${SRC_DIR}/cpu_gemm.c")
LIST(REMOVE_ITEM SRC_FILES "${SRC_DIR}/darknet.c")

if (CUDA_FOUND)

FILE(GLOB CU_SRC_FILES ${SRC_DIR}/*.cu)
    
CUDA_INCLUDE_DIRECTORIES( ${SRC_DIR} )

set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${CU_SRC_FILES})

    
message(STATUS "CUDA FOUND")
endif()


# -------- LIBRARY CREATION ---------
set( LIB_TYPE "SHARED" )
#set( LIB_TYPE "STATIC" )

if( APPLE )
    message( STATUS "Building ${LIB_TYPE} for APPLE" )
    set( LIB_NAME           darknetOSX )
    set( LIB_PATH_INSTALL  ${PROJECT_SOURCE_DIR}/../lib/osx/ )
elseif( LINUX )
    message( STATUS "Building ${LIB_TYPE} for LINUX" )
    set( LIB_NAME           darknetLinux )
    set( LIB_PATH_INSTALL  ${PROJECT_SOURCE_DIR}/../lib/linux/ )
endif()


add_library(            ${LIB_NAME} ${LIB_TYPE} ${SRC_FILES} ${CU_SRC_FILES} ${cuda_objs})
target_link_libraries(  ${LIB_NAME}        ${LIBRARIES} )


set( LIB_PATH  ${LIB_PATH_INSTALL} )
set_target_properties(  ${LIB_NAME}
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY    ${LIB_PATH}
        ARCHIVE_OUTPUT_DIRECTORY    ${LIB_PATH}
        )

# -------- EXECUTABLE CREATION (NOT OPENFRAMEWORKS) ---------
#add_executable(darknet ${SRC_DIR}/darknet.c)
#target_link_libraries(darknet ${LIBRARIES} ${LIB_NAME})
#target_link_libraries(darknet ${LIB_NAME})
