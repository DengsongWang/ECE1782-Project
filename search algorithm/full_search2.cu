#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WIDTH 352 //1920
#define HEIGHT 288 //1080
#define SEARCH_RANGE 8

__device__  const int RANGE_8[288][2] = {
    {-1, 0}, {0, -1}, {0, 1}, {1, 0}, {-2, 0}, {-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, -1}, {1, 1}, {2, 0}, {-3, 0}, {-2, -1}, {-2, 1}, {-1, -2},
    {-1, 2}, {0, -3}, {0, 3}, {1, -2}, {1, 2}, {2, -1}, {2, 1}, {3, 0}, {-4, 0}, {-3, -1}, {-3, 1}, {-2, -2}, {-2, 2}, {-1, -3}, {-1, 3}, {0, -4}, {0, 4},
    {1, -3}, {1, 3}, {2, -2}, {2, 2}, {3, -1}, {3, 1}, {4, 0}, {-5, 0}, {-4, -1}, {-4, 1}, {-3, -2}, {-3, 2}, {-2, -3}, {-2, 3}, {-1, -4}, {-1, 4}, {0, -5},
    {0, 5}, {1, -4}, {1, 4}, {2, -3}, {2, 3}, {3, -2}, {3, 2}, {4, -1}, {4, 1}, {5, 0}, {-6, 0}, {-5, -1}, {-5, 1}, {-4, -2}, {-4, 2}, {-3, -3}, {-3, 3},
    {-2, -4}, {-2, 4}, {-1, -5}, {-1, 5}, {0, -6}, {0, 6}, {1, -5}, {1, 5}, {2, -4}, {2, 4}, {3, -3}, {3, 3}, {4, -2}, {4, 2}, {5, -1}, {5, 1}, {6, 0},
    {-7, 0}, {-6, -1}, {-6, 1}, {-5, -2}, {-5, 2}, {-4, -3}, {-4, 3}, {-3, -4}, {-3, 4}, {-2, -5}, {-2, 5}, {-1, -6}, {-1, 6}, {0, -7}, {0, 7}, {1, -6}, {1, 6},
    {2, -5}, {2, 5}, {3, -4}, {3, 4}, {4, -3}, {4, 3}, {5, -2}, {5, 2}, {6, -1}, {6, 1}, {7, 0}, {-8, 0}, {-7, -1}, {-7, 1}, {-6, -2}, {-6, 2}, {-5, -3},
    {-5, 3}, {-4, -4}, {-4, 4}, {-3, -5}, {-3, 5}, {-2, -6}, {-2, 6}, {-1, -7}, {-1, 7}, {0, -8}, {0, 8}, {1, -7}, {1, 7}, {2, -6}, {2, 6}, {3, -5}, {3, 5},
    {4, -4}, {4, 4}, {5, -3}, {5, 3}, {6, -2}, {6, 2}, {7, -1}, {7, 1}, {8, 0}, {-8, -1}, {-8, 1}, {-7, -2}, {-7, 2}, {-6, -3}, {-6, 3}, {-5, -4}, {-5, 4},
    {-4, -5}, {-4, 5}, {-3, -6}, {-3, 6}, {-2, -7}, {-2, 7}, {-1, -8}, {-1, 8}, {1, -8}, {1, 8}, {2, -7}, {2, 7}, {3, -6}, {3, 6}, {4, -5}, {4, 5}, {5, -4},
    {5, 4}, {6, -3}, {6, 3}, {7, -2}, {7, 2}, {8, -1}, {8, 1}, {-8, -2}, {-8, 2}, {-7, -3}, {-7, 3}, {-6, -4}, {-6, 4}, {-5, -5}, {-5, 5}, {-4, -6}, {-4, 6},
    {-3, -7}, {-3, 7}, {-2, -8}, {-2, 8}, {2, -8}, {2, 8}, {3, -7}, {3, 7}, {4, -6}, {4, 6}, {5, -5}, {5, 5}, {6, -4}, {6, 4}, {7, -3}, {7, 3}, {8, -2},
    {8, 2}, {-8, -3}, {-8, 3}, {-7, -4}, {-7, 4}, {-6, -5}, {-6, 5}, {-5, -6}, {-5, 6}, {-4, -7}, {-4, 7}, {-3, -8}, {-3, 8}, {3, -8}, {3, 8}, {4, -7}, {4, 7},
    {5, -6}, {5, 6}, {6, -5}, {6, 5}, {7, -4}, {7, 4}, {8, -3}, {8, 3}, {-8, -4}, {-8, 4}, {-7, -5}, {-7, 5}, {-6, -6}, {-6, 6}, {-5, -7}, {-5, 7}, {-4, -8},
    {-4, 8}, {4, -8}, {4, 8}, {5, -7}, {5, 7}, {6, -6}, {6, 6}, {7, -5}, {7, 5}, {8, -4}, {8, 4}, {-8, -5}, {-8, 5}, {-7, -6}, {-7, 6}, {-6, -7}, {-6, 7},
    {-5, -8}, {-5, 8}, {5, -8}, {5, 8}, {6, -7}, {6, 7}, {7, -6}, {7, 6}, {8, -5}, {8, 5}, {-8, -6}, {-8, 6}, {-7, -7}, {-7, 7}, {-6, -8}, {-6, 8}, {6, -8},
    {6, 8}, {7, -7}, {7, 7}, {8, -6}, {8, 6}, {-8, -7}, {-8, 7}, {-7, -8}, {-7, 8}, {7, -8}, {7, 8}, {8, -7}, {8, 7}, {-8, -8}, {-8, 8}, {8, -8}, {8, 8}
};

__device__ int calculateSAD(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                            int x_idx, int y_idx, int x_ref, int y_ref) {
    int sad = 0;
    for (int i = 0; i < BLOCK_SIZE; i++) {
        for (int j = 0; j < BLOCK_SIZE; j++) {
            int curr_pixel = d_curr_frame[(y_idx + j) * WIDTH + (x_idx + i)];
            int ref_pixel = d_ref_frame[(y_ref + j) * WIDTH + (x_ref + i)];
            sad += abs(curr_pixel - ref_pixel);
        }
    }
    return sad;
}
__global__ void diamond_search(unsigned char *d_curr_frame, unsigned char *d_ref_frame,
                                    int *d_motion_vectors, int* d_sad_list) {
    int y_block = blockIdx.x * blockDim.x + threadIdx.x; // row
    int x_block = blockIdx.y * blockDim.y + threadIdx.y; // col

    int x_idx = x_block * BLOCK_SIZE;
    int y_idx = y_block * BLOCK_SIZE;

    int right_bound = WIDTH - BLOCK_SIZE;
    int bottom_bound = HEIGHT - BLOCK_SIZE;
    int row_block_num = WIDTH / BLOCK_SIZE;

    if (x_idx <= right_bound && y_idx <= bottom_bound) {
        int best_sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_idx, y_idx);
        // motion vertor is relative distance
        int best_x = 0;
        int best_y = 0;

        // full search
        int search_size = SEARCH_RANGE * 2 + 1; // 17
        int search_num = search_size * search_size - 1; // 288
        for (int i = 0; i <= search_num; i++) {
            int x_ref = x_idx + RANGE_8[i][1];
            int y_ref = y_idx + RANGE_8[i][0];

            if (x_ref >= 0 && x_ref <= right_bound && y_ref >= 0 && y_ref <= bottom_bound) {
                int sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_ref, y_ref);
                
                if(y_block * row_block_num + x_block == 77) {
                    //printf("%d %d %d\n", i, j, sad);
                }

                if (sad < best_sad) {
                    best_sad = sad;
                    best_x = RANGE_8[i][1];
                    best_y = RANGE_8[i][0];
                }
            }
        }

        // Store the motion vector
        
        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
        d_sad_list[block_idx] = best_sad;

        if(y_block * row_block_num + x_block == 77) {
            //printf("%d %d %d\n", best_y, best_x, best_sad);
            //printf("%d %d %d\n", block_idx, d_motion_vectors[block_idx * 2], d_motion_vectors[block_idx * 2 + 1]);
        }
    }
}

int* block_match_diamond_single_frame(unsigned char* h_curr_frame, unsigned char* h_ref_frame) {
    int *h_motion_vectors;
    int *h_sad_list;

    unsigned char *d_curr_frame;
    unsigned char *d_ref_frame;
    int *d_motion_vectors;
    int *d_sad_list;

    int x_block_num = WIDTH / BLOCK_SIZE;
    int y_block_num = HEIGHT / BLOCK_SIZE;

    size_t frame_size = WIDTH * HEIGHT * sizeof(unsigned char);
    size_t sad_size = x_block_num * y_block_num * sizeof(int);

    h_motion_vectors = (int *)malloc(2 * sad_size);
    h_sad_list = (int *)malloc(sad_size);

    cudaMalloc((void **)&d_curr_frame, frame_size);
    cudaMalloc((void **)&d_ref_frame, frame_size);
    cudaMalloc((void **)&d_motion_vectors, 2 * sad_size);
    cudaMalloc((void **)&d_sad_list, sad_size);

    cudaMemcpy(d_curr_frame, h_curr_frame, frame_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref_frame, h_ref_frame, frame_size, cudaMemcpyHostToDevice);

    // Blocks configuration
    dim3 grid(8, 8);
    dim3 block((y_block_num + grid.x - 1) / grid.x, (x_block_num + grid.y - 1) / grid.y);

    // Launch the diamond search kernel
    diamond_search<<<block, grid>>>(d_curr_frame, d_ref_frame, d_motion_vectors, d_sad_list);

    cudaDeviceSynchronize();

    cudaMemcpy(h_motion_vectors, d_motion_vectors, 2 * sad_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sad_list, d_sad_list, sad_size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_curr_frame);
    cudaFree(d_ref_frame);
    cudaFree(d_motion_vectors);
    cudaFree(d_sad_list);

    free(h_sad_list);

    return h_motion_vectors;
}

unsigned char* readIntArrayFromFile(const char* filename, int x, int y) {
    int i, j, temp;
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return NULL;
    }

    // Allocate memory for the unsigned char array
    unsigned char* array = (unsigned char*)malloc(x * y * sizeof(unsigned char));
    if (array == NULL) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }

    for (i = 0; i < y; i++) {
        for (j = 0; j < x; j++) {
            if (fscanf(file, "%d", &temp) == 1) {
                // Ensure the value is within the 0-255 range
                if (temp >= 0 && temp <= 255) {
                    array[i * x + j] = (unsigned char)temp;
                } else {
                    fprintf(stderr, "Value out of range: %d\n", temp);
                    free(array);
                    fclose(file);
                    return NULL;
                }
            } else {
                fprintf(stderr, "Failed to read an integer\n");
                free(array);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return array;
}

int* read2(const char* filename, int x, int y) {
    int i, j, temp;
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return NULL;
    }

    // Allocate memory for the int array
    int* array = (int*)malloc(x * y * sizeof(int));
    if (array == NULL) {
        perror("Failed to allocate memory");
        fclose(file);
        return NULL;
    }

    for (i = 0; i < y; i++) {
        for (j = 0; j < x; j++) {
            if (fscanf(file, "%d", &temp) == 1) {
                // Ensure the value is within the 0-255 range
                array[i * x + j] = temp;
            } else {
                fprintf(stderr, "Failed to read an integer\n");
                free(array);
                fclose(file);
                return NULL;
            }
        }
    }

    fclose(file);
    return array;
}

int main() {
    unsigned char *h_curr_frame = readIntArrayFromFile("frame2_origin.txt", WIDTH, HEIGHT);
    unsigned char *h_ref_frame = readIntArrayFromFile("frame1_reconst.txt", WIDTH, HEIGHT);
    int *motion_vectors = block_match_diamond_single_frame(h_curr_frame, h_ref_frame);

    int mv_length = WIDTH * HEIGHT / BLOCK_SIZE / BLOCK_SIZE;
    int *mv_cpu = read2("frame2_mv.txt", 2, mv_length);

    bool flag = 1;
    for(int i = 0; i < mv_length * 2; i++) {
        if(mv_cpu[i] != motion_vectors[i]){
            flag = 0;
            printf("mv index %d does not pair\n", i);
            break;
        }
    }

    if(flag) printf("mv pairs\n");
    //printf("%d\n",mv_cpu[23]);
    
    //save into file
    FILE* file = fopen("mv.txt", "w");
    size_t mv_size = (WIDTH / BLOCK_SIZE) * (HEIGHT / BLOCK_SIZE);
    for (size_t i = 0; i < mv_size; i++) {
        fprintf(file, "%d %d\n", motion_vectors[2 * i], motion_vectors[2 * i + 1]);
    }
    fclose(file);

    FILE* file1 = fopen("mv_cpu.txt", "w");
    for (size_t i = 0; i < mv_size; i++) {
        fprintf(file1, "%d %d\n", mv_cpu[2 * i], mv_cpu[2 * i + 1]);
    }
    fclose(file1);


    return 0;
    
}
