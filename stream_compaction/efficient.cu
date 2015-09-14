#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

#define blockSize 128
int *temp_scan;
int *scan_result;

__global__ void upSweep(int n, int d, int *o_data, int *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;	
	if (index <= n) {
		if (index % (int)pow(2.0, d+1) == 0) {
			o_data[index-1] = i_data[index - 1 - (int)pow(2.0, d)] + i_data[index - 1];
		} 
	}
}

__global__ void downSweep(int n, int d, int *o_data, int *i_data) {
	int index =  (blockIdx.x * blockDim.x) + threadIdx.x;
	int temp = 0;
	if (index <= n) {
		if (index % (int)pow(2.0, d+1) == 0) {
			temp = i_data[index - 1 - (int)pow(2.0, d)];
			o_data[index - 1 - (int)pow(2.0, d)] = i_data[index-1];
			o_data[index-1] = temp + i_data[index - 1];
		} 
	}

}

void scan(int n, int *odata, const int *idata) {
    int d = ilog2ceil(n);
	int total = (int) pow(2.0, d);

	cudaMalloc((void**)&scan_result, total * sizeof(int));
	cudaMalloc((void**)&temp_scan, total * sizeof(int));
	
	cudaMemcpy(temp_scan, idata, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scan_result, idata, total * sizeof(int), cudaMemcpyHostToDevice);

	dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);
	
	for (int i = 0; i < d; i++) {
		upSweep<<<fullBlocksPerGrid, blockSize>>>(total, i, scan_result, temp_scan);
		temp_scan = scan_result;
	}

	
	cudaMemcpy(odata, scan_result, total * sizeof(int), cudaMemcpyDeviceToHost);
	odata[total-1] = 0;

	cudaMemcpy(scan_result, odata, total * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(temp_scan, odata, total * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = d-1; i >= 0; i--) {
		downSweep<<<fullBlocksPerGrid, blockSize>>>(total, i, scan_result, temp_scan);
		temp_scan = scan_result;
	}

	cudaMemcpy(odata, scan_result, total * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(scan_result);
	cudaFree(temp_scan);

}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
    // TODO
    return -1;
}

}
}
