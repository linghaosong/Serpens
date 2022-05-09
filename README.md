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


To learn more about the techinqual details, please see [this link](https://arxiv.org/abs/2109.11081).


If you find this code useful, please cite:

    @inproceedings{song2022serpens,
    author = {Linghao Song and Yuze Chi and Atefeh Sohrabizadeh and Young-kyu Choi and Jason Lau and Jason Cong},
    title = {Sextans: A Streaming Accelerator for General-Purpose Sparse-Matrix Dense-Matrix Multiplication},
    booktitle={2022 59th ACM/IEEE Design Automation Conference (DAC)},
    year = {2022}
    }
