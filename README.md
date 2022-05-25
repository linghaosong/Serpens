[![DOI](https://zenodo.org/badge/480584531.svg)](https://zenodo.org/badge/latestdoi/480584531)

# Serpens

Serpens is an HBM FPGA accelerator for SpMV. 

Dependencies: 
1. TAPA + Autobridge

+ Following [Install TAPA](https://tapa.readthedocs.io/en/release/installation.html) to install TAPA(Autobridge) and Gurobi.
+ Vitis 2021.2
+ Xilinx xilinx_u280_xdma_201920_3 shell and a Xilinx U280 FPGA card.

### Input matrix format & sample input
The host code takes martrix market format(https://math.nist.gov/MatrixMarket/formats.html). We test on sparse matrices from SuiteSparse(https://sparse.tamu.edu) collection. To download a sample matrix for running, do

    mkdir matrices
    sh download_example.sh
    cmake ..
    make swsim
    
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

    TAPAB=../bitstream_a16/Serpens_xilinx_u280_xdma_201920_3.xclbin ./serpens ../matrices/nasa4704/nasa4704.mtx


To learn more about the techinqual details, please see [this link](https://arxiv.org/abs/2111.12555).


If you find this code useful, please cite:

    @inproceedings{song2022serpens,
    author = {Linghao Song and Yuze Chi and Licheng Guo and Jason Cong},
    title = {Serpens: A High Bandwidth Memory Based Accelerator for General-Purpose Sparse Matrix-Vector Multiplication},
    booktitle={2022 59th ACM/IEEE Design Automation Conference (DAC)},
    year = {2022}
    }
