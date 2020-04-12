cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cutw LANGUAGES CXX CUDA)
enable_testing()
find_package(Threads)

include_directories(src)
include_directories(transwarp/src)

add_library(cutw
    src/cutw/copy.cu
    src/cutw/copy.h
    src/cutw/DeviceArray.cu
    src/cutw/DeviceArray.h
    src/cutw/error.cu
    src/cutw/error.h
    src/cutw/HostArray.cu
    src/cutw/HostArray.h
    src/cutw/RandomGenerator.cu
    src/cutw/RandomGenerator.h
    src/cutw/Stream.cu
    src/cutw/Stream.h
    src/cutw/tasks.cpp
    src/cutw/tasks.h
    src/cutw/transwarp.h
)

add_executable(cutw_test
    test/test.cpp
    test/catch.hpp
)

target_compile_features(cutw PRIVATE cxx_std_11)
target_compile_features(cutw_test PRIVATE cxx_std_11)
target_link_libraries(cutw_test cutw curand)

add_test(cutw_test cutw_test --use-colour no)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /bigobj /EHsc /wd4503 /wd4996 /wd4702 /wd4100")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wconversion -Wno-unused-but-set-parameter -Wno-unused-parameter")
	if (CMAKE_COMPILER_IS_GNUCC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	endif()
    target_link_libraries(cutw ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(cutw_test ${CMAKE_THREAD_LIBS_INIT})
endif()
