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
#define SCAN_MAX_THREADS_PER_BLOCK 32
#define ELEMENTS_PER_BLOCK SCAN_MAX_THREADS_PER_BLOCK * 2

// TODO get more info about this
// There were two BCAO optimisations in the paper - this one is fastest
#define SCAN_CONFLICT_FREE_OFFSET(n) ((n) >> SCAN_LOG_MEM_BANKS)
/*////////////////////////////////////////*/

/*/////////// Some consts for reduction //////////////*/
#define REDUCE_MAX_THREADS_PER_BLOCK 128
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

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
  if (blockSize >= 64)
    sdata[tid] += sdata[tid + 32];
  if (blockSize >= 32)
    sdata[tid] += sdata[tid + 16];
  if (blockSize >= 16)
    sdata[tid] += sdata[tid + 8];
  if (blockSize >= 8)
    sdata[tid] += sdata[tid + 4];
  if (blockSize >= 4)
    sdata[tid] += sdata[tid + 2];
  if (blockSize >= 2)
    sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void small_reduce2D(const int *g_idata, int *g_odata, int num_rows,
                               int num_cols) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y; // Each row gets its own set of blocks
  unsigned int i = row * num_cols + blockIdx.x * (blockSize * 2) + tid;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  // Initialize shared memory
  sdata[tid] = 0;

  // Load data into shared memory for this block's chunk of the row
  while (i < (row + 1) * num_cols) {
    sdata[tid] +=
        g_idata[i] +
        (i + blockSize < (row + 1) * num_cols ? g_idata[i + blockSize] : 0);
    i += gridSize;
  }
  __syncthreads();

  // Perform the reduction within the block (similar to reduce6)
  if (blockSize >= 512) {
    if (tid < 256)
      sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (tid < 128)
      sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (tid < 64)
      sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  if (tid < 32)
    warpReduce<blockSize>(sdata, tid);

  // Store partial results for each row
  if (tid == 0) {
    atomicAdd(&g_odata[row],
              sdata[0]); // Combine partial sums from all blocks of the row
  }
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
    printf("small scan");
    // small scan
    small_2d_scan<<<blocks, threads_per_block_per_col,
                    2 * threads_per_block_per_col * sizeof(int)>>>(out_2d,
                                                                   in_2d);
  } else {
    printf("large scan");

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
  int threads_per_row = cols / 2;
  unsigned int threads_per_block_per_row;
  if (threads_per_row <= REDUCE_MAX_THREADS_PER_BLOCK) {
    threads_per_block_per_row = threads_per_row;
  } else {
    threads_per_block_per_row = REDUCE_MAX_THREADS_PER_BLOCK;
    while (threads_per_row % threads_per_block_per_row != 0) {
      threads_per_block_per_row /= 2;
    }
  }

  int blocks_per_row = threads_per_row / threads_per_block_per_row;
  int blocks = rows * blocks_per_row;

  dim3 grid(blocks_per_row, rows); // 2D grid: (blocks per row, number of rows)
  dim3 block(
      threads_per_block_per_row); // 1D block with calculated threads per block

  // Calculate shared memory size per block
  int shared_mem_size = threads_per_block_per_row * sizeof(int);

  if (blocks_per_row == 1) {
    printf("small reduce");
    // small reduce
    small_reduce2D<REDUCE_MAX_THREADS_PER_BLOCK>
        <<<grid, block, shared_mem_size>>>(in_2d, out, rows, cols);
  } else {
    // TODO not working

    printf("large reduce");
    // large reduce
    int *sums;
    cudaMalloc((void **)&sums, blocks * sizeof(sums[0]));

    small_reduce2D<SCAN_MAX_THREADS_PER_BLOCK>
        <<<grid, block, shared_mem_size>>>(in_2d, sums, rows, cols);

    reduce_2d_arr(out, sums, blocks_per_row, rows);

    cudaFree(sums);
  }
}

__global__ void copy_first_column(int *output, int *input, int clients,
                                  int periods) {
  int threadID = threadIdx.x;

  output[threadID] = input[threadID * clients];
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {

  // calc_clients_accounts(account, changes, clients, periods);
  prefix_sum_2d_arr_per_cols(account, changes, clients, periods);

  // if cudalast error print the error
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
  }
}
