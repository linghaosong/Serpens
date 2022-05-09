cmake_minimum_required(VERSION 3.14)

project(serpens)

set(PLATFORM
    xilinx_u280_xdma_201920_3
    CACHE STRING "Target FPGA platform")

set(TOP Serpens)

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-write-strings")

find_package(TAPA REQUIRED)
find_package(SDx REQUIRED)

add_executable(serpens)
target_sources(serpens PRIVATE src/serpens-host.cpp src/serpens.cpp)
target_link_libraries(serpens PRIVATE tapa::tapa)

add_tapa_target(
  hls
  --run-floorplanning
  --enable-hbm-binding-adjustment
  --floorplan-opt-priority SLR_CROSSING_PRIORITIZED
  INPUT src/serpens.cpp
  TOP ${TOP}
  CONNECTIVITY ${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
  CONSTRAINT ${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  --read-only-args edge_list_ptr
  --read-only-args edge_list_ch*
  --read-only-args vec_X
  --read-only-args vec_Y
  --write-only-args vec_Y_out
  PLATFORM ${PLATFORM})

add_xocc_hw_link_targets(
  ${CMAKE_CURRENT_BINARY_DIR}
  --config=${CMAKE_CURRENT_SOURCE_DIR}/link_config.ini
  --vivado.synth.jobs 8
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=1
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=EarlyBlockPlacement
  --vivado.prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore
  --vivado.prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=${CMAKE_CURRENT_BINARY_DIR}/constraint.tcl
  INPUT hls
  HW_EMU_XCLBIN hw_emu_xclbin
  HW_XCLBIN hw_xclbin)

add_custom_target(
  swsim
  COMMAND $<TARGET_FILE:serpens> ../matrices/nasa4704/nasa4704.mtx
  DEPENDS serpens
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hwsim
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_emu_xclbin},FILE_NAME> $<TARGET_FILE:serpens>
          /share/slh/suitesparse/rgg010/rgg010.mtx
  DEPENDS serpens ${hw_emu_xclbin}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(
  hw
  COMMAND sh ${CMAKE_CURRENT_BINARY_DIR}/${TOP}.${PLATFORM}.hw_generate_bitstream.sh
  COMMAND TAPAB=$<TARGET_PROPERTY:${hw_xclbin},FILE_NAME> $<TARGET_FILE:serpens>
          /share/slh/suitesparse/rgg010/rgg010.mtx
  DEPENDS sextans hls
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
