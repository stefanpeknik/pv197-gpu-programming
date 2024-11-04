/**
 * @brief Kernel function to solve the problem on the GPU.
 *
 * This function is expected to perform computations on the GPU using the
 * provided input data. It processes the changes and updates the account and sum
 * arrays accordingly.
 *
 * @param changes Pointer to an array of changes to be applied. It is a 2D array
 * where each "row" represents a client and each "column" represents a period.
 * Each element represents the change in the account balance for a client in a
 * period.
 * @param account Pointer to an array representing the account balances. It is a
 * 2D array where each "row" represents a client and each "column" represents a
 * period. Each element represents the account balance for a client in a period.
 * @param sum Pointer to an array representing the sum of account balances for
 * each period. It is a 1D array where each element represents the sum of
 * account balances for a period.
 * @param clients The number of clients. It is the number of rows in the
 * changes, account, and sum arrays.
 * @param periods The number of periods. It is the number of columns in the
 * changes, account, and sum arrays.
 *
 * The function does not return any value. It modifies the account and sum
 * arrays in place.
 */

/*/////////////Some consts////////////////*/
// TODO dicide on specific consts
#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
#define MAX_THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK MAX_THREADS_PER_BLOCK * 2

// TODO get more info about this
// There were two BCAO optimisations in the paper - this one is fastest
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
/*////////////////////////////////////////*/

/*////////////////TODO my stuff///////////*/
void cp_from_device_and_print(int *d_arr, int length) {
  int *arr = (int *)malloc(length * sizeof(int));
  cudaMemcpy(arr, d_arr, length * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < length; i++) {
    printf("%d ", arr[i]);
  }
  printf("\n");
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

/*////////cuda prefix sum/////////////////*/

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
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

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
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

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
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
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
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

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
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

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
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

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
    sums[sums_index] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
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
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

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

/*///////////scan funcs//////////////*/

/*///////////////////////////////////*/

void prefix_sum_2d_arr_per_cols(int *out_2d, int *in_2d, int cols, int rows) {
  int threads_per_col = rows / 2;
  int threads_per_block_per_col;
  if (threads_per_col <= MAX_THREADS_PER_BLOCK) {
    threads_per_block_per_col = threads_per_col;
  } else {
    threads_per_block_per_col = MAX_THREADS_PER_BLOCK;
    while (threads_per_col % threads_per_block_per_col != 0) {
      threads_per_block_per_col /= 2;
    }
  }
  int blocks_per_col = threads_per_col / threads_per_block_per_col;
  int blocks = cols * blocks_per_col;

  // printf("threads_per_col: %d\n", threads_per_col);
  // printf("threads_per_block_per_col: %d\n", threads_per_block_per_col);
  // printf("blocks_per_col: %d\n", blocks_per_col);
  // printf("blocks: %d\n", blocks);

  if (blocks_per_col == 1) {

    // small scan
    small_2d_scan<<<blocks, threads_per_block_per_col,
                    2 * threads_per_block_per_col * sizeof(int)>>>(out_2d,
                                                                   in_2d);
  } else {

    int *sums, *incr;
    cudaMalloc((void **)&sums, blocks * sizeof(sums[0]));
    cudaMalloc((void **)&incr, blocks * sizeof(incr[0]));

    // large scan
    large_2d_scan<<<blocks, threads_per_block_per_col,
                    2 * threads_per_block_per_col * sizeof(int)>>>(
        out_2d, in_2d, blocks_per_col, sums);

    // printf("out2d in between: ");
    // cp_from_device_and_print(out_2d, cols * rows);

    // printf("sums: ");
    // cp_from_device_and_print(sums, blocks);

    // scan appropriately sums
    prefix_sum_2d_arr_per_cols(incr, sums, cols, blocks_per_col);

    // printf("incr: ");
    // cp_from_device_and_print(incr, blocks);

    // add sums to out_2d
    incr_out_with_incr<<<blocks, threads_per_block_per_col * 2>>>(
        out_2d, incr, blocks_per_col);

    // printf("out_2d: \n");
    // cp_from_device_and_print(out_2d, cols * rows);

    cudaFree(sums);
    cudaFree(incr);
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
}
