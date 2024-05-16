__global__ void optimizedDiamondSearchKernel(unsigned char *currentFrame, 
    unsigned char *referenceFrame, int *motionVectors, int width, int height) {
    int blockX = blockIdx.x * BLOCK_SIZE;
    int blockY = blockIdx.y * BLOCK_SIZE;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    // Calculate the starting point for this thread in the block
    int startX = blockX + (threadId % BLOCK_SIZE);
    int startY = blockY + (threadId / BLOCK_SIZE);

    if (startX < width && startY < height) {
        int bestSAD = INT_MAX;
        int bestX = 0;
        int bestY = 0;

        // Assuming LDS and SDS patterns are applied here

        // Each thread checks a specific point or small region rather than the whole block
        for (int i = 0; i < 8; i++) { // For LDS
            int refX = startX + LDS_PATTERN[i][0];
            int refY = startY + LDS_PATTERN[i][1];

            if (refX >= 0 && refX < width && refY >= 0 && refY < height) {
                int sad = calculateSAD(currentFrame, referenceFrame, startX, startY, refX, refY, width);
                if (sad < bestSAD) {
                    bestSAD = sad;
                    bestX = refX;
                    bestY = refY;
                }
            }
        }

        // Similar process for SDS

        // Store the motion vector for this particular point or region
        motionVectors[startY * width + startX] = bestX - startX;
        motionVectors[startY * width + startX + 1] = bestY - startY;
    }
}
