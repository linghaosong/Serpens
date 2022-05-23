tapac \
  --work-dir run \
  --top Serpens \
  --part-num xcu280-fsvh2892-2L-e \
  --platform xilinx_u280_xdma_201920_3 \
  --clock-period 3.33 \
  -o Serpens.xo \
  --constraint Serpens_floorplan.tcl \
  --connectivity ../link_config_a16.ini \
  --read-only-args edge_list_ptr \
  --read-only-args edge_list_ch* \
  --read-only-args vec_X \
  --read-only-args vec_Y \
  --write-only-args vec_Y_out \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --enable-hbm-binding-adjustment \
  --run-floorplan-dse \
  --min-area-limit 0.6 \
  --min-slr-width-limit 5000 \
  ../src/serpens.cpp \
   2>&1 | tee tapa.log
