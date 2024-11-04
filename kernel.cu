/**
 * REDUCTION IS REALLY PRIMITIVE AND NOT OPTIMIZED
 * scan is kind of fine but bank offsets are not taken into account yet
 *
 * source of info for the prefix sum algorithm:
 * https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf
 * https://github.com/mattdean1/cuda
 *
 * source of info for the reduction algorithm:
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 *
 */

#define SCAN_THREADS_PER_BLOCK 512

#define REDUCE_THREADS_PER_BLOCK 128

/*////////////////TODO my stuff///////////*/
void cp_from_device_and_print(int *d_arr, int rows, int cols) {
  int *arr = (int *)malloc(rows * cols * sizeof(int));
  cudaMemcpy(arr, d_arr, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", arr[i * cols + j]);
    }
    printf("\n");
  }
  free(arr);
}
/*////////////////////////////////////////*/

__global__ void incr_out_with_incr(int *output, int *incr) {
  if (blockIdx.x == 0) {
    return;
  }

  int row_offset = blockIdx.x * (gridDim.y * blockDim.x) + blockIdx.y;
  // blockIdx.x - offset to the correct blockx (row)
  // * (gridDim.y * blockDim.x) - but have to take into out2d the length
  // of the previous blocks
  // + blockIdx.y - offset to the correct blocky (column)
  int output_index = row_offset + threadIdx.x * gridDim.y;
  int incr_index = (blockIdx.x - 1) * gridDim.y +
                   blockIdx.y; // -1 because we need to skip the first block

  output[output_index] += incr[incr_index];
}

__global__ void prefix_sum_cols(int *in2d, int *out2d, int *sums) {
  extern __shared__ int temp[];

  int els_per_block = blockDim.x * 2;

  int row_offset = blockIdx.x * (gridDim.y * els_per_block) + blockIdx.y;
  // blockIdx.x - offset to the correct blockx (row)
  // * (gridDim.y * els_per_block) - but have to take into out2d the length
  // of the previous blocks where also each handles 2 elements
  // + blockIdx.y - offset to the correct blocky (column)
  int ai = threadIdx.x;
  int bi = threadIdx.x + (els_per_block / 2);
  int ai_input = row_offset + ai * gridDim.y;
  int bi_input = row_offset + bi * gridDim.y;
  temp[ai] = in2d[ai_input];
  temp[bi] = in2d[bi_input];

  int offset = 1;
  for (int d = els_per_block >> 1; d > 0;
       d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    sums[blockIdx.x * gridDim.y + blockIdx.y] = temp[els_per_block - 1];
    temp[els_per_block - 1] = 0;
  }

  for (int d = 1; d < els_per_block; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadIdx.x < d) {
      int ai = offset * (2 * threadIdx.x + 1) - 1;
      int bi = offset * (2 * threadIdx.x + 2) - 1;

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  ai_input = row_offset + ai * gridDim.y;
  bi_input = row_offset + bi * gridDim.y;
  out2d[ai_input] = temp[ai] + in2d[ai_input];
  out2d[bi_input] = temp[bi] + in2d[bi_input];
}

__global__ void reduce_rows(int *in2d, int *out2d) {
  extern __shared__ int temp[];

  int els_per_block = blockDim.y * 2;

  int offset =
      blockIdx.x * (gridDim.y * els_per_block) + blockIdx.y * els_per_block;

  unsigned int tid = threadIdx.x;
  unsigned int i = offset + tid;

  temp[tid] = in2d[i];
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      temp[tid] += temp[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    out2d[blockIdx.x * gridDim.y + blockIdx.y] = temp[0];
}

void prefix_sum_2d_per_col(int *in2d, int *out2d, int cols, int rows) {
  int y = cols;
  int threads_per_client = rows / 2;
  int threads_per_block;
  if (threads_per_client <= SCAN_THREADS_PER_BLOCK) {
    threads_per_block = threads_per_client;
  } else {
    threads_per_block = SCAN_THREADS_PER_BLOCK;
    while (threads_per_client % threads_per_block != 0) {
      threads_per_block /= 2;
    }
  }

  dim3 grid;
  grid.x = threads_per_client / threads_per_block;
  grid.y = y;
  grid.z = 1;

  int *sums;
  int blocks = grid.x * grid.y;
  cudaMalloc((void **)&sums, blocks * sizeof(sums[0]));

  prefix_sum_cols<<<grid, threads_per_block,
                    2 * SCAN_THREADS_PER_BLOCK * sizeof(int)>>>(in2d, out2d,
                                                                sums);

  if (grid.x > 1) {
    int *incr;
    cudaMalloc((void **)&incr, blocks * sizeof(incr[0]));

    prefix_sum_2d_per_col(sums, incr, grid.y, grid.x);

    incr_out_with_incr<<<grid, threads_per_block * 2>>>(out2d, incr);
  }
}

// TODO: implement
void reduce_2d_per_row(int *in2d, int *out2d, int cols, int rows) {
  if (cols == 1) {
    return;
  }

  // ???????????????????????????????????????????
  int x = rows;
  int y = cols / REDUCE_THREADS_PER_BLOCK;

  dim3 grid;
  grid.x = x;
  grid.y = y;
  grid.z = 1;

  int *sums;
  cudaMalloc((void **)&sums, x * y * sizeof(sums[0]));
  reduce_rows<<<grid, REDUCE_THREADS_PER_BLOCK,
                REDUCE_THREADS_PER_BLOCK * sizeof(int)>>>(in2d, out2d);

  reduce_2d_per_row(out2d, sums, y, x);
  // ???????????????????????????????????????????
}

// temp before finishing the proper reduction
__global__ void reduce_temp(int *account, int *sum, int clients, int periods) {
  int periodId = blockIdx.x * blockDim.x + threadIdx.x;

  if (periodId < periods) {
    int s = 0;

    for (int i = 0; i < clients; i++) {
      s += account[periodId * clients + i];
    }

    sum[periodId] = s;
  }
}
void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  prefix_sum_2d_per_col(changes, account, clients, periods);

  reduce_2d_per_row(account, sum, clients, periods);

  // dim3 sumBlocks(periods / REDUCE_THREADS_PER_BLOCK);
  // reduce_temp<<<sumBlocks, REDUCE_THREADS_PER_BLOCK>>>(account, sum, clients,
  //                                                      periods);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
  }
}