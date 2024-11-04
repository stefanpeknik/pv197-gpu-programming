/**
 * source of info for the prefix sum algorithm:
 * https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf
 * https://github.com/mattdean1/cuda
 *
 * source of info for the reduction algorithm:
 * https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
 */

/*///////////// Some consts for scan////////////////*/
// TODO dicide on specific consts
#define SCAN_SHARED_MEMORY_BANKS 32
#define SCAN_LOG_MEM_BANKS 5
#define SCAN_MAX_THREADS_PER_BLOCK                                             \
  2 // I have some memory issues with anything
    // above 2, so it needs debugging
#define ELEMENTS_PER_BLOCK SCAN_MAX_THREADS_PER_BLOCK * 2

// TODO get more info about this
// There were two BCAO optimisations in the paper - this one is fastest
#define SCAN_CONFLICT_FREE_OFFSET(n) ((n) >> SCAN_LOG_MEM_BANKS)
/*////////////////////////////////////////*/

/*/////////// Some consts for reduction //////////////*/
#define REDUCE_MAX_THREADS_PER_BLOCK 256
/*////////////////////////////////////////*/

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

/*//////////////cuda stuff////////////////*/

__global__ void incr_out_with_incr(int *output, int *incr, int blocks_per_col) {
  int cols = gridDim.x / blocks_per_col;
  int current_col = blockIdx.x / blocks_per_col;
  int col_blockId = blockIdx.x % blocks_per_col;

  // col_bockId is in respect to output, meaning we skip there the first block
  // and for incr we need to skip the last block, so we move one block back
  // (see the sum_index) calculation

  if (col_blockId <= 0) {
    return;
  }

  int threads_per_block = blockDim.x;
  int els_per_block = threads_per_block; // Each thread handles two
  int row_offset = col_blockId * els_per_block;

  int threadID = threadIdx.x;

  // calc offsetted indices for the 2d input array but also for colunms
  int out_index = (row_offset + threadID) * cols + current_col;
  int sum_index = (col_blockId - 1) * cols + current_col;

  output[out_index] += incr[sum_index];
}

/*////////////////////////////////////////*/

/*/////// cuda prefix sum /////////////////*/

// Kernel function to perform a large prefix sum (scan) operation

__global__ void small_2d_scan(int *output, int *input) {
  // Shared memory for temporary storage
  extern __shared__ int temp[];

  int n = blockDim.x * 2;

  // Calculate block and thread indices
  int threadID = threadIdx.x;

  // Calculate indices for the elements this thread will work on
  int ai = threadID;
  int bi = threadID + (n / 2);

  // calc offsetted indices for the 2d input array but also for colunms
  int input_ai = ai * gridDim.x + blockIdx.x;
  int input_bi = bi * gridDim.x + blockIdx.x;

  // Calculate bank offsets to avoid shared memory bank conflicts
  int bankOffsetA = SCAN_CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = SCAN_CONFLICT_FREE_OFFSET(bi);

  // Load input elements into shared memory with bank conflict avoidance
  temp[ai + bankOffsetA] = input[input_ai];
  temp[bi + bankOffsetB] = input[input_bi];

  // Offset for the reduction step
  int offset = 1;

  // Build sum in place up the tree - up-sweep phase
  for (int d = n >> 1; d > 0; d >>= 1) {
    // Synchronize threads to ensure all data is loaded
    __syncthreads();
    if (threadID < d) {
      // Calculate indices for the reduction step
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;

      // Apply bank offsets
      ai += SCAN_CONFLICT_FREE_OFFSET(ai);
      bi += SCAN_CONFLICT_FREE_OFFSET(bi);

      // Perform the reduction
      temp[bi] += temp[ai];
    }
    // Double the offset for the next level of the tree
    offset *= 2;
  }

  // Synchronize threads before moving to the next phase
  __syncthreads();

  // Save the total sum of this block to the sums array and clear the last
  // element
  if (threadID == 0) {
    temp[n - 1 + SCAN_CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  // Traverse down the tree and build the scan - down-sweep phase
  for (int d = 1; d < n; d *= 2) {
    // Halve the offset for the next level of the tree
    offset >>= 1;
    // Synchronize threads before the next step
    __syncthreads();
    if (threadID < d) {
      // Calculate indices for the scan step
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;

      // Apply bank offsets
      ai += SCAN_CONFLICT_FREE_OFFSET(ai);
      bi += SCAN_CONFLICT_FREE_OFFSET(bi);

      // Perform the scan operation
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  // Synchronize threads before writing the output
  __syncthreads();

  // Write the results to the output array
  input_ai = ai * gridDim.x + blockIdx.x;
  input_bi = bi * gridDim.x + blockIdx.x;
  output[input_ai] = temp[ai + bankOffsetA] + input[input_ai];
  output[input_bi] = temp[bi + bankOffsetB] + input[input_bi];
}

__global__ void large_2d_scan(int *output, int *input, int blocks_per_col,
                              int *sums) {
  // Shared memory for temporary storage
  extern __shared__ int temp[];

  // TODO this for sure needs to get optimized with 2d grid
  int cols = gridDim.x / blocks_per_col;
  int current_col = blockIdx.x / blocks_per_col;
  int col_blockId = blockIdx.x % blocks_per_col;
  int threads_per_block = blockDim.x;
  int els_per_block = threads_per_block * 2; // Each thread handles two
  int row_offset = col_blockId * els_per_block;

  int threadID = threadIdx.x;
  int n = els_per_block;

  // Calculate indices for the elements this thread will work on
  int ai = threadID;
  int bi = threadID + (n / 2);

  // calc offsetted indices for the 2d input array but also for colunms
  int input_ai = (row_offset + ai) * cols + current_col;
  int input_bi = (row_offset + bi) * cols + current_col;

  // Calculate bank offsets to avoid shared memory bank conflicts
  int bankOffsetA = SCAN_CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = SCAN_CONFLICT_FREE_OFFSET(bi);

  // TODO remove
  //  print col, blockincol, threadid, ai, bi, input_ai, input_bi,
  printf("col: %d, blockincol: %d, threadid: %d, ai: %d, bi: %d, input_ai: %d, "
         "input_bi: %d\n",
         current_col, col_blockId, threadID, ai, bi, input_ai, input_bi);

  // Load input elements into shared memory with bank conflict avoidance
  temp[ai + bankOffsetA] = input[input_ai];
  temp[bi + bankOffsetB] = input[input_bi];

  // Offset for the reduction step
  int offset = 1;

  // Build sum in place up the tree - up-sweep phase
  for (int d = n >> 1; d > 0; d >>= 1) {
    // Synchronize threads to ensure all data is loaded
    __syncthreads();
    if (threadID < d) {
      // Calculate indices for the reduction step
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;

      // Apply bank offsets
      ai += SCAN_CONFLICT_FREE_OFFSET(ai);
      bi += SCAN_CONFLICT_FREE_OFFSET(bi);

      // Perform the reduction
      temp[bi] += temp[ai];
    }
    // Double the offset for the next level of the tree
    offset *= 2;
  }

  // Synchronize threads before moving to the next phase
  __syncthreads();

  // Save the total sum of this block to the sums array and clear the last
  // element
  if (threadID == 0) {
    int sums_index = col_blockId * cols + current_col;
    sums[sums_index] = temp[n - 1 + SCAN_CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + SCAN_CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  // Traverse down the tree and build the scan - down-sweep phase
  for (int d = 1; d < n; d *= 2) {
    // Halve the offset for the next level of the tree
    offset >>= 1;
    // Synchronize threads before the next step
    __syncthreads();
    if (threadID < d) {
      // Calculate indices for the scan step
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;

      // Apply bank offsets
      ai += SCAN_CONFLICT_FREE_OFFSET(ai);
      bi += SCAN_CONFLICT_FREE_OFFSET(bi);

      // Perform the scan operation
      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }

  // Synchronize threads before writing the output
  __syncthreads();

  // Write the results to the output array
  input_ai = (row_offset + ai) * cols + current_col;
  input_bi = (row_offset + bi) * cols + current_col;
  output[input_ai] = temp[ai + bankOffsetA] + input[input_ai];
  output[input_bi] = temp[bi + bankOffsetB] + input[input_bi];
}

/*///////////////////////////////////*/

/*/////////// reduction //////////////*/

__global__ void reduce(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];
  // each thread loads one element from global to shared mem
  int row = blockIdx.y;
  int col = blockIdx.x;
  int cols = gridDim.x;
  int tid = threadIdx.x;
  int i = row * cols + col * (blockDim.x * 2) + tid;

  sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

/*///////////////////////////////////*/

void prefix_sum_2d_arr_per_cols(int *out_2d, int *in_2d, int cols, int rows) {
  printf("\n\n");
  printf("scan in_2d: \n");
  cp_from_device_and_print(in_2d, rows, cols);

  int threads_per_col = rows / 2;
  int threads_per_block_per_col;
  if (threads_per_col <= SCAN_MAX_THREADS_PER_BLOCK) {
    threads_per_block_per_col = threads_per_col;
  } else {
    threads_per_block_per_col = SCAN_MAX_THREADS_PER_BLOCK;
    while (threads_per_col % threads_per_block_per_col != 0) {
      threads_per_block_per_col /= 2;
    }
  }
  int blocks_per_col = threads_per_col / threads_per_block_per_col;
  int blocks = cols * blocks_per_col;

  printf("threads_per_col: %d\n", threads_per_col);
  printf("threads_per_block_per_col: %d\n", threads_per_block_per_col);
  printf("blocks_per_col: %d\n", blocks_per_col);
  printf("blocks: %d\n", blocks);

  if (blocks_per_col == 1) {
    printf("small scan\n");
    // small scan
    small_2d_scan<<<blocks, threads_per_block_per_col,
                    2 * threads_per_block_per_col * sizeof(int)>>>(out_2d,
                                                                   in_2d);
  } else {
    printf("large scan\n");

    int *sums, *incr;
    cudaMalloc((void **)&sums, blocks * sizeof(sums[0]));
    cudaMalloc((void **)&incr, blocks * sizeof(incr[0]));

    // large scan
    large_2d_scan<<<blocks, threads_per_block_per_col,
                    2 * threads_per_block_per_col * sizeof(int)>>>(
        out_2d, in_2d, blocks_per_col, sums);

    printf("out2d in between: \n");
    cp_from_device_and_print(out_2d, rows, cols);

    printf("sums: \n");
    cp_from_device_and_print(sums, 1, blocks);

    // scan appropriately sums
    prefix_sum_2d_arr_per_cols(incr, sums, cols, blocks_per_col);

    printf("incr: \n");
    cp_from_device_and_print(incr, 1, blocks);

    // add sums to out_2d
    incr_out_with_incr<<<blocks, threads_per_block_per_col * 2>>>(
        out_2d, incr, blocks_per_col);

    printf("out_2d: \n");
    cp_from_device_and_print(out_2d, rows, cols);

    cudaFree(sums);
    cudaFree(incr);
  }
}

void reduce_2d_arr(int *out, int *in_2d, int cols, int rows) {
  // create a grid of size ((cols + REDUCE_MAX_THREADS_PER_BLOCK - 1) /
  // REDUCE_MAX_THREADS_PER_BLOCK) x rows and a block of size
  // REDUCE_MAX_THREADS_PER_BLOCK x 1

  printf("\n\nreduce\n");

  int threads_per_col = rows / 2;
  int threads_per_block_per_col;
  if (threads_per_col <= REDUCE_MAX_THREADS_PER_BLOCK) {
    threads_per_block_per_col = threads_per_col;
  } else {
    threads_per_block_per_col = REDUCE_MAX_THREADS_PER_BLOCK;
    while (threads_per_col % threads_per_block_per_col != 0) {
      threads_per_block_per_col /= 2;
    }
  }
  int blocks_per_col = threads_per_col / threads_per_block_per_col;
  int blocks = cols * blocks_per_col;

  printf("threads_per_col: %d\n", threads_per_col);
  printf("threads_per_block_per_col: %d\n", threads_per_block_per_col);
  printf("blocks_per_col: %d\n", blocks_per_col);
  printf("blocks: %d\n", blocks);

  if (blocks_per_col == 1) {
    reduce<<<cols, threads_per_block_per_col,
             threads_per_block_per_col * sizeof(int)>>>(out, in_2d);
  } else {
    int *sums;
    cudaMalloc((void **)&sums, blocks * sizeof(sums[0]));

    // recurse till only one remains
    reduce_2d_arr(sums, in_2d, cols, blocks_per_col);
  }
}

__global__ void copy_first_column(int *output, int *input, int clients,
                                  int periods) {
  int threadID = threadIdx.x;

  output[threadID] = input[threadID * clients];
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  prefix_sum_2d_arr_per_cols(account, changes, clients, periods);
  reduce_2d_arr(sum, account, clients, periods);

  printf("\n\n-------------\n");

  printf("account: \n");
  cp_from_device_and_print(account, periods, clients);

  printf("sum: \n");
  cp_from_device_and_print(sum, 1, periods);

  // if cudalast error print the error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
  }
}
