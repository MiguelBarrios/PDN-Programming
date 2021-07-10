__global__ void combineIntArr(int* X, int y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        int update = X[i] + y * X[i]
        atomicAdd(&y, update);
    }
}


__global__ void dotProductMultiplicationPart(float* A, float* B, float* C,int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}


#define BLOCK_SIZE 512
//size = number of elements 
__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    __shared__ float in_s[BLOCK_SIZE];
    int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    in_s[threadIdx.x] = ((idx              < size)? in[idx]:            0.0f) +
		 	((idx + BLOCK_SIZE < size)? in[idx+BLOCK_SIZE]: 0.0f);

    for(int stride = BLOCK_SIZE / 2; stride >= 1; stride = stride / 2) {
	__syncthreads();
	if(threadIdx.x < stride)
		in_s[threadIdx.x] += in_s[threadIdx.x + stride];
    }

    if(threadIdx.x == 0)
	out[blockIdx.x] = in_s[0];
}
