#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WIDTH 352 //1920
#define HEIGHT 288 //1080
#define SEARCH_RANGE 16

// Define the diamond search pattern
__device__  const int LDS_PATTERN[8][2] = {{0, -2}, {1, -1}, {2, 0}, {1, 1}, {0, 2}, {-1, 1}, {-2, 0}, {-1, -1}};
__device__  const int SDS_PATTERN[4][2] = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
//#define LDS_PATTERN[8][2] {{0, -2}, {1, -1}, {2, 0}, {1, 1}, {0, 2}, {-1, 1}, {-2, 0}, {-1, -1}}
//#define SDS_PATTERN[4][2] {{0, -1}, {1, 0}, {0, 1}, {-1, 0}}

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
    int y_block = blockIdx.x * blockDim.x + threadIdx.x;
    int x_block = blockIdx.y * blockDim.y + threadIdx.y;

    int x_idx = x_block * BLOCK_SIZE;
    int y_idx = y_block * BLOCK_SIZE;

    int right_bound = WIDTH - BLOCK_SIZE;
    int bottom_bound = HEIGHT - BLOCK_SIZE;

    if (x_idx < right_bound && y_idx < bottom_bound) {
        int best_sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_idx, y_idx);
        // motion vertor is relative distance
        int best_x = 0;
        int best_y = 0;

        // Large Diamond Search (LDS)
        while(true) {
            bool flag = 1;
            int best_x_temp = 0;
            int best_y_temp = 0;

            for (int i = 0; i < 8; i++) {
                int x_dis = best_x + LDS_PATTERN[i][0];
                int y_dis = best_x + LDS_PATTERN[i][1];
                int x_ref = x_idx + x_dis;
                int y_ref = y_idx + y_dis;

                if (abs(x_dis) <= SEARCH_RANGE && abs(y_dis) <= SEARCH_RANGE
                && x_ref >= 0 && x_ref < right_bound && y_ref >= 0 && y_ref < bottom_bound) {
                    int sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_ref, y_ref);
                    if (sad < best_sad) {
                        best_sad = sad;
                        best_x_temp = x_dis;
                        best_y_temp = y_dis;
                        flag = 0;
                    }
                }
            }
            if(flag) break;

            best_x = best_x_temp;
            best_y = best_y_temp;
        }
        
        /*
        00100
        01210
        12321
        01210
        00100
        */

        int best_sds_x = 0;
        int best_sds_y = 0;

        // Small Diamond Search (SDS)
        for (int i = 0; i < 4; i++) {
            int x_dis = best_x + SDS_PATTERN[i][0];
            int y_dis = best_y + SDS_PATTERN[i][1];
            int x_ref = x_idx + x_dis;
            int y_ref = y_idx + y_dis;

            if (abs(x_dis) + abs(y_dis) <= SEARCH_RANGE && x_ref >= 0
                    && x_ref < right_bound && y_ref >= 0 && y_ref < bottom_bound) {
                int sad = calculateSAD(d_curr_frame, d_ref_frame, x_idx, y_idx, x_ref, y_ref);
                if (sad < best_sad) {
                    best_sad = sad;
                    best_sds_x = SDS_PATTERN[i][0];
                    best_sds_y = SDS_PATTERN[i][1];
                }
            }
        }

        best_x += best_sds_x;
        best_y += best_sds_y;

        // Store the motion vector
        int row_block_num = WIDTH / BLOCK_SIZE;
        int block_idx = y_block * row_block_num + x_block;
        d_motion_vectors[block_idx * 2] = best_y;
        d_motion_vectors[block_idx * 2 + 1] = best_x;
        d_sad_list[block_idx] = best_sad;
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
    int count = 0;
    for(int i = 0; i < mv_length * 2; i++) {
        if(mv_cpu[i] != motion_vectors[i]){
            count++;
            printf("%d %d\n", i/2, i%2);
        }
    }
    printf("not match num: %d", count);

    //if(flag) printf("mv pairs\n");
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
