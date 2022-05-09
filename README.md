# Serpens

Serpens is an HBM FPGA accelerator for SpMV. 

Dependencies: 
1. TAPA + Autobridge

+ Following [Install TAPA](https://tapa.readthedocs.io/en/release/installation.html) to install TAPA(Autobridge) and Gurobi.
+ Vitis 2021.2
+ Xilinx xilinx_u280_xdma_201920_3 shell and a Xilinx U280 FPGA card.


## To do software emulation:

    mkdir build
    cd build
    cmake ..
    make swsim
    
## To run HLS:

    cd build
    cp ../run_tapa.sh ./
    sh run_tapa.sh

After HLS, a bitstream generator file *Serpens_generate_bitstream.sh* is generated under the build folder. 

## To generate bitstream (hardware):

    sh Serpens_generate_bitstream.sh
    
## To run the accelerator on board:
We provide the generated bitstream. If you have a U280 FPGA ready, you don't need to generate the hwardware again, just run

    TAPAB=../bitstream/Serpens_xilinx_u280_xdma_201920_3.xclbin ./serpens ../matrices/nasa4704/nasa4704.mtx
