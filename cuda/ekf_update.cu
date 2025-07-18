extern "C" {
__global__ void kalman_update(float* x, float* P, const float* gps, float* K_out) {
    int i = threadIdx.x;

    __shared__ float H[8];
    if (i == 0) H[0] = 1.0f; // H(0,0)
    if (i == 1) H[5] = 1.0f; // H(1,1)

    if (i < 2) {
        K_out[i*4 + i] = 0.5f; // Dummy gain
        x[i] = x[i] + K_out[i*4 + i] * (gps[i] - x[i]);
    }

    if (i < 16) {
        P[i] *= 0.9f;
    }
}

// Wrapper function for use in .cpp files
void launch_kalman_update(float* x, float* P, const float* gps, float* K_out) {
    kalman_update<<<1, 4>>>(x, P, gps, K_out);
}
}
