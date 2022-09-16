# Sparse Matrix Vector Multiplication
## Run Instruction
`g++ spmspv.cpp -std=c++11 -pthread -o spmspv`

then by openning spmspv file, the performance results will be save in a csv file.
You cna see results visualized by running visualize.ipynb, which use csv file to draw charts.

## Methods:
`COO`, `CSS`, `CSR`, `ELL`, `COO-parallel`, `COO-parallel`, `CSR-parallel`, `ELL-parallel`, and `CSR-simd`
