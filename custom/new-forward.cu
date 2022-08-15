#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define MAX_N_THREADS 1024
#define BSegSize0 500
// #define TILE_LONG 32
__constant__ float Kc[16 * 4 * 7 * 7];

/*1. Tiled shared memory convolution*/
__global__ void conv_forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    y - output
    x - input
    k - kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = y4d(0,0,0,0)
    // y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
// #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define Kc4d(i3, i2, i1, i0) Kc[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    const int X_tile_width = TILE_WIDTH + K - 1;
    const int W_grid = ceil((float)W_out / TILE_WIDTH);
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];

#define X_shared2d(i1, i0) X_shared[(i1) * (X_tile_width) + i0]
// #define K_shared2d(i1, i0) K_shared2d[(i1) * (K) + i0]

    int b = blockIdx.x;
    int m = blockIdx.y;
    int h0 = threadIdx.x;
    int w0 = threadIdx.y;
    int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
    int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
    int h = h_base + h0;
    int w = w_base + w0;
    float res = 0;
    for(int c = 0; c < C; ++c){
        for(int ii = h; ii < h_base + X_tile_width; ii += TILE_WIDTH){
            for(int jj = w; jj < w_base + X_tile_width; jj += TILE_WIDTH){
                if(ii < H  && jj < W){
                    X_shared2d(ii - h_base, jj - w_base) = x4d(b, c, ii, jj);
                }
            }
        }
        __syncthreads();
        if(h < H_out && w < W_out){
            for(int p = 0; p < K; ++p){
                for(int q = 0; q < K; ++q){
                    res += X_shared2d(h0 + p, w0 + q) * Kc4d(m, c, p, q);
                }
            }
        }
        __syncthreads();
    }
    if(h < H_out && w < W_out){
        y4d(b, m, h, w) = res;
    }
#undef y4d
#undef x4d
// #undef k4d
#undef X_shared2d
// #undef K_shared2d
}

/*2. Shared memory matrix multiplication and input matrix unrolling*/
__global__ void conv_forward_mult_mat_unroll_kernel(float *y, const float *x_unroll, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;
    const int H_unroll = C * K * K;
    // const int TILE_LONG = (int)ceil((float)MAX_N_THREADS / M);
#define x_unroll3d(i2, i1, i0) x_unroll[(i2) * (H_unroll * W_unroll) + (i1) * (W_unroll) + i0]

    // Insert your GPU convolution kernel code here
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];
    float* K_shared = &shmem[M * M];

#define X_shared2d(i1, i0) X_shared[(i1) * (M) + i0]
#define K_shared2d(i1, i0) K_shared[(i1) * (M) + i0]

    
    int b = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int Xunroll_col_base = blockIdx.z * M;
    // int Xunroll_row_base = (blockIdx.z / W_grid) * TILE_LONG;
    int Xunroll_col = Xunroll_col_base + ty;
    int Xunroll_row0 = tx;
    int Kunroll_row = tx;
    int Kunroll_col0 = ty;
    int Out_col = Xunroll_col;
    int Out_row = Kunroll_row;
    float res = 0;
    int Xunroll_row;
    int Kunroll_col;
    for(int jj = 0; jj < ceil((float)H_unroll / M); jj++){
        Kunroll_col = jj * M + Kunroll_col0;
        if(Kunroll_col < H_unroll){
            K_shared2d(tx, ty) = k[Kunroll_row * C * K * K + Kunroll_col];
        } else {
            K_shared2d(tx, ty) = 0;
        }
        Xunroll_row = jj * M + Xunroll_row0;
        if(Xunroll_col < W_unroll && Xunroll_row < H_unroll){
            X_shared2d(tx, ty) = x_unroll3d(b, Xunroll_row, Xunroll_col);
        } else {
            X_shared2d(tx, ty) = 0;
        }
        __syncthreads();
        // if(Out_row < M && Out_col < W_unroll){
        for(int ii = 0; ii < M; ii++){
            res += K_shared2d(Out_row, ii) * X_shared2d(ii, ty);
        }
        // }
        __syncthreads();
    }
    if(Out_row < M && Out_col < W_unroll){
        y[b * M * W_unroll + Out_row * W_unroll + Out_col] = res;
    }

#undef x_unroll3d
#undef X_shared2d
#undef K_shared2d
}
	
/*2. Shared memory matrix multiplication and input matrix unrolling*/
__global__ void unroll_kernel(int B, int C, int H, int W, int K, const float* x, float* x_unroll){
    int b, c, s, h_out, w_out, h_unroll, w_base, p, q;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;
    b = blockIdx.y;
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll3d(i2, i1, i0) x_unroll[(i2) * (C * K * K * W_unroll) + (i1) * (W_unroll) + i0]
    // float dumb;
    if(t < C * W_unroll){
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_base = c * K * K;
        h_unroll = s;
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++){
                if(h_out + p < H && w_out + q < W){
                    x_unroll3d(b, w_base + p * K + q, h_unroll) = x4d(b, c, h_out + p, w_out + q);
                } else {
                    x_unroll3d(b, w_base + p * K + q, h_unroll) = 0;
                }
            }
        }
    }
#undef x4d
#undef x_unroll3d
}

/*3. Kernel fusion for unrolling and matrix-multiplication*/
__global__ void conv_forward_fused_unroll_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K){
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    const int W_unroll = H_out * W_out;
    const int H_unroll = C * K * K;
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // Insert your GPU convolution kernel code here
    extern __shared__ float shmem[];
    float* X_shared = &shmem[0];

#define X_shared2d(i1, i0) X_shared[(i1) * (M) + i0]
// #define K_shared2d(i1, i0) K_shared[(i1) * (H_unroll) + i0]
#define Kc2d(i1, i0) Kc[(i1) * (H_unroll) + i0]

    int b = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int s, p, q;
    int Xunroll_col_base = blockIdx.y * M;
    int Xunroll_col = Xunroll_col_base + ty;
    int c;
    int Xunroll_row = tx;
    // int Kunroll_col = ty;
    int h_out = Xunroll_col / W_out;
    int w_out = Xunroll_col % W_out;
    if(Xunroll_col < W_unroll){
        while(Xunroll_row < H_unroll){
            c = Xunroll_row / (K * K);
            s = Xunroll_row % (K * K);
            p = s / K;
            q = s % K;
            X_shared2d(Xunroll_row, ty) = x4d(b, c, h_out + p, w_out + q);
            Xunroll_row += M;
        }
    } else {
        while(Xunroll_row < H_unroll){
            X_shared2d(Xunroll_row, ty) = 0;
            Xunroll_row += M;
        }
    }

    // while(Kunroll_col < H_unroll){
    //     K_shared2d(tx, Kunroll_col) = k[tx * H_unroll + Kunroll_col];
    //     Kunroll_col += M;
    // }

    __syncthreads();
    
    int Out_col = Xunroll_col;
    float res = 0;
    for(int jj = 0; jj < ceil((float)H_unroll / M); jj++){
        // if(Out_col < W_unroll){
        for(int ii = jj * M; ii < jj * M + M && ii < H_unroll; ii++){
            res += Kc2d(tx, ii) * X_shared2d(ii, ty);
        }
        // }
    }
    if(Out_col < W_unroll){
        y[b * M * W_unroll + tx * W_unroll + Out_col] = res;
    }

#undef X_shared2d
#undef Kc2d
#undef x4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_y, const float *host_x, const float *host_k, float **device_y_ptr, float **device_x_ptr, float **device_k_ptr, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // get_device_properties();
    // int BSegSize;
    // if(B < 2 * BSegSize0){
    //     BSegSize = B / 2;
    // } else {
    //     BSegSize = BSegSize0;
    // }
    // cudaMalloc((void**)device_x_ptr, sizeof(float*) * 1);
    // cudaMalloc((void**)device_y_ptr, sizeof(float) * (BSegSize * 2 * M * H_out * W_out + BSegSize * 2 * C * H * W));
    cudaMalloc((void**)device_x_ptr, sizeof(float*) * B * C * H * W);
    cudaMalloc((void**)device_y_ptr, sizeof(float) * B * M * H_out * W_out);
    cudaMalloc((void**)device_k_ptr, sizeof(float) * M * C * K * K);
    // cudaMemset(*device_y_ptr, 0, sizeof(float) * BSegSize * 2 * M * H_out * W_out);
    cudaMemset(*device_y_ptr, 0, sizeof(float) * B * M * H_out * W_out);
    // const float* a[2];
    // a[0] = host_y;
    // a[1] = host_x;
    // cudaMemcpy(*device_y_ptr, a, sizeof(float*) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_x_ptr, host_x, sizeof(float) * B * C * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, sizeof(float) * M * C * K * K, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"conv_forward_gpu_prolog(): CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_y, const float *device_x_, const float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    cudaError_t error;

    const float *device_x = device_x_;

    /*1. Tiled shared memory convolution*/
    /*4. Weight matrix (kernel values) in constant memory*/
    cudaMemcpyToSymbol(Kc, device_k, C * K * K * M * sizeof(float));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(B, M, ceil((float)W_out/TILE_WIDTH) * ceil((float)H_out/TILE_WIDTH));
    size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1));
    conv_forward_kernel<<<dimGrid, dimBlock, shmem_size>>>(device_y, device_x, B, M, C, H, W, K);

    /*2. Shared memory matrix multiplication and input matrix unrolling*/
    // float* device_x_unroll;
    // dim3 dimBlock(M, M, 1);
    // size_t shmem_size = sizeof(float) * 2 * M * M;
    // int B_ = (B > 1000) ? 1000 : B;
    // dim3 dimGrid;
    // cudaMalloc((void**)&device_x_unroll, sizeof(float) * B_ * C * K * K * H_out * W_out);
    // int cnt = 0;
    // while(cnt < B){
    //     B_ = ((B - cnt) > 1000) ? 1000 : (B - cnt);
    //     dimGrid = dim3(B_, 1, ceil(H_out * W_out/(float)M));
    //     unroll_kernel<<<dim3(ceil(C * H_out * W_out / (float)MAX_N_THREADS), B_, 1), MAX_N_THREADS>>>(B_, C, H, W, K, device_x + cnt * C * H * W, device_x_unroll);
    //     cudaDeviceSynchronize(); 
    //     conv_forward_mult_mat_unroll_kernel<<<dimGrid, dimBlock, shmem_size>>>(device_y + cnt * M * H_out * W_out, device_x_unroll, device_k, B_, M, C, H, W, K);
    //     cudaDeviceSynchronize();
    //     cnt += B_;
    // }
    // cudaFree(device_x_unroll);

    /*3. Kernel fusion for unrolling and matrix-multiplication*/
    /*4. Weight matrix (kernel values) in constant memory*/
    // cudaMemcpyToSymbol(Kc, device_k, C * K * K * M * sizeof(float));
    // dim3 dimBlock(M, M, 1);
    // dim3 dimGrid(B, ceil(H_out * W_out/(float)M), 1);
    // size_t shmem_size = sizeof(float) * C * K * K * M;
    // conv_forward_fused_unroll_kernel<<<dimGrid, dimBlock, shmem_size>>>(device_y, device_x, B, M, C, H, W, K);
    // cudaDeviceSynchronize();


    /*5. Using Streams to overlap computation with data transfer*/
    // const int XSegSize = C * H * W;
    // const int YSegSize = M * H_out * W_out;
    // float* hosts[2];
    // cudaMemcpy(hosts, device_y, sizeof(float*) * 2, cudaMemcpyDeviceToHost);
    // float* host_y = hosts[0];
    // float* host_x = hosts[1];
    // int BSegSize;
    // if(B < 2 * BSegSize0){
    //     BSegSize = B / 2;
    // } else {
    //     BSegSize = BSegSize0;
    // }
    // float* device_x = device_y + 2 * BSegSize * YSegSize;
    // float* device_y1 = device_y + BSegSize * YSegSize;
    // float* device_x1 = device_x + BSegSize * XSegSize;
    // cudaStream_t stream0, stream1;
    // cudaStreamCreate(&stream0);
    // cudaStreamCreate(&stream1);
    // dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 dimGrid(BSegSize, M, ceil((float)W_out/TILE_WIDTH) * ceil((float)H_out/TILE_WIDTH));
    // cudaMemcpyToSymbol(Kc, device_k, C * K * K * M * sizeof(float));
    // size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1));
    // for(int b = 0; b < B; b += 2 * BSegSize){
    //     cudaMemcpyAsync(device_x, host_x + b * XSegSize, sizeof(float) * BSegSize * XSegSize, cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(device_x1, host_x + (b + BSegSize) * XSegSize, sizeof(float) * BSegSize * XSegSize, cudaMemcpyHostToDevice, stream1);
    //     conv_forward_kernel<<<dimGrid, dimBlock, shmem_size, stream0>>>(device_y, device_x, B, M, C, H, W, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, shmem_size, stream1>>>(device_y1, device_x1, B, M, C, H, W, K);
    //     cudaMemcpyAsync(host_y + b * YSegSize, device_y, sizeof(float) * BSegSize * YSegSize, cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_y + (b + BSegSize) * YSegSize, device_y1, sizeof(float) * BSegSize * YSegSize, cudaMemcpyDeviceToHost, stream1);
    // }
    // cudaStreamDestroy(stream0);
    // cudaStreamDestroy(stream1);

    // Useful snippet for error checking
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"conv_forward_gpu(): CUDA error: 2 "<<B <<" " <<M<<" " <<C<<" " << H << " " << W <<" "<< K<< " " <<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y, float *device_x, float *device_k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    // Copy the output back to host
    cudaMemcpy(host_y, device_y, sizeof(float) * B * M * H_out * W_out, cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_k);
    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"conv_forward_gpu_epilog()5: CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
