# Statement
This file is the code for the final project of the ECE 1782 course. This is a group collaboration project, and the members of the group are:

Runheng Lu, 
Yiming Zheng, 
Jingsheng Zhang, 
Dengsong Wang

# Project Overview

This project contains CUDA scripts for Motion Estimation in video encoding. The project is organized into several directories, each containing different approaches and versions of the algorithms.

## Key Files

- `best_approach.cu`: The best performing approach.
- `util_motion.cu`: Utility functions for the best performing appoarch.

## Directory Structure

- `GPUapproach1/`: Contains the first approach to the problem, CUDA Thread-Per-Block Matching.
- `GPUapproach2/`: Contains the second approach to the problem, CUDA Block-Per-Block Motion Vector Matching.
- `CPUversion/`: Contains the CPU version of the algorithm and scripts for generating images.
- `search algorithm/`: Contains different search algorithms implemented in CUDA.
- `util_motion.cu`: Contains utility functions for motion detection.


## Building and Running

Compile only best_approach.cu. The input file is the extended version of foreman.yuv(y only file) link: https://media.xiph.org/video/derf/

