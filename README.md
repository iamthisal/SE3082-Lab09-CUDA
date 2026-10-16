# SE3082 Lab 09 – Introduction to CUDA (Google Colab Edition)

**Student ID:** IT24101868
**Module:** SE3082 – Parallel Computing, Semester 2, 2026

## Run in Google Colab
1. Open `SE3082_Lab09_IT24101868.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type → T4 GPU → Save**.
3. Choose **Runtime → Run all**. The notebook writes, compiles and runs every CUDA program.
4. When finished, choose **Runtime → Disconnect and delete runtime** to release the GPU.

## Environment (Exercise 1)
- GPU: Tesla T4, 15360 MiB memory
- Compute capability: 7.5 (compiler flag `-arch=sm_75`)
- Driver version: 580.82.07, CUDA version 13.0
- CUDA compiler: release 13.0, V13.0.88 (`nvcc --version`)
- `hello.cu` printed four "Hello from GPU thread" lines (threads 0 to 3, in any order)

## Files
| File | Description |
|---|---|
| `SE3082_Lab09_IT24101868.ipynb` | Submission notebook with all exercises and outputs |
| `hello.cu` | Smoke test: 1 block of 4 threads |
| `ex2_add.cu` | Adds two integers on the GPU (1 block, 1 thread) |
| `ex3_blocks.cu` | Vector addition, 512 blocks × 1 thread |
| `ex4_threads.cu` | Vector addition, 1 block × 512 threads |
| `ex5_vecmul.cu` | Multiplies two 10,000,000-element vectors, 512 threads per block, 19,532 blocks, with bounds guard |
| `ex6_matrix.cu` | Element-wise product of two 10,000 × 10,000 matrices, 32 × 16 threads per block, 313 × 625 blocks, with bounds guard on both dimensions |

## Results
- Exercise 2: `Result is 9`
- Exercise 3: 0 mismatches out of 512
- Exercise 4: 0 mismatches out of 512
- Exercise 5: 0 mismatches out of 10000000
- Exercise 6: 0 mismatches out of 100000000

All programs were compiled with `-arch=sm_75`.
