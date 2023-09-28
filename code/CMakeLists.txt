cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II 9.3.0 REQUIRED
    HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
    )
deal_ii_initialize_cached_variables()

project(waveEquation)

add_executable(main main.cc include/waveMath.h)
deal_ii_setup_target(main)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)
