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
__global__ void add(int *output, int length, int *n) {
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * length;

  output[blockOffset + threadID] += n[blockID];
}

/*////////////////////////////////////////*/

/*////////cuda prefix sum/////////////////*/

// Kernel function to perform a large prefix sum (scan) operation

__global__ void scan(int *output, int *input, int n, int *sums) {
  // Shared memory for temporary storage
  extern __shared__ int temp[];

  // Calculate block and thread indices
  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  // Calculate indices for the elements this thread will work on
  int ai = threadID;
  int bi = threadID + (n / 2);

  // Calculate bank offsets to avoid shared memory bank conflicts
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  // Load input elements into shared memory with bank conflict avoidance
  temp[ai + bankOffsetA] = input[blockOffset + ai];
  temp[bi + bankOffsetB] = input[blockOffset + bi];

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
    sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
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
  output[blockOffset + ai] = temp[ai + bankOffsetA] + input[blockOffset + ai];
  output[blockOffset + bi] = temp[bi + bankOffsetB] + input[blockOffset + bi];
}

/*///////////////////////////////////*/

/*///////////scan funcs//////////////*/

/*///////////////////////////////////*/

__global__ void calc_single_client_acc(int *account, int *changes, int clients,
                                       int periods) {
  int client = blockIdx.x * blockDim.x + threadIdx.x;

  if (periods <= ELEMENTS_PER_BLOCK) {
    int threads = periods / 2;
    int *sums;
    cudaMalloc((void **)&sums, sizeof(sums[0])); // TODO remove this
    scan<<<1, threads, periods * sizeof(int)>>>(account, changes, periods,
                                                sums);
    cudaFree(sums);
  } else {
  }
}

void calc_clients_accounts(int *account, int *changes, int clients,
                           int periods) {
  int clients_per_block = MAX_THREADS_PER_BLOCK;
  while (!(clients % clients_per_block == 0 && clients_per_block > 128)) {
    clients_per_block -= 2;
  }
  int blocks = clients / clients_per_block;

  calc_single_client_acc<<<blocks, clients_per_block>>>(account, changes,
                                                        clients, periods);
}

void calc_accounts(int *account, int *changes, int clients, int periods) {
  int client_num_of_periods = periods;
  if (client_num_of_periods > ELEMENTS_PER_BLOCK) {
    // perform a large scan (meaning more than one block per client)
    // to distinguish the clients blocks from each other we define 2Dim grid
    // where the x axis is the client and the y axis is the block
    int remainder = client_num_of_periods % ELEMENTS_PER_BLOCK;
    // if we can fit the clients periods exactly in the blocks with max
    // elements per block
    int blocks_per_client = client_num_of_periods / ELEMENTS_PER_BLOCK;
    dim3 numBlocks(clients, blocks_per_client);
    dim3 threadsPerBlock(ELEMENTS_PER_BLOCK);
    // else
    // option 1
    if (remainder != 0) {
      // find a number of elements per block (meaning must be <= than
      // ELEMENTS_PER_BLOCK) that can divide the number of periods that way we
      // ensure that each block is full and each client has the same number of
      // blocks

      // find the number of elements per block
      int elements_per_block = 0;
      for (int i = ELEMENTS_PER_BLOCK; i > 0; i--) {
        if (client_num_of_periods % i == 0) {
          elements_per_block = i;
          break;
        }
      }

      // find the number of blocks per client
      blocks_per_client = client_num_of_periods / elements_per_block;
      dim3 numBlocks(clients, blocks_per_client);
      dim3 threadsPerBlock(elements_per_block);
    }

    // TODO scan large <<<numBlocks, threadsPerBlock>>>
  } else {
    // only need one block to scan each client
    // so here we use a block for each client and set the number of threads to
    // the number of periods
    dim3 numBlocks(clients);
    dim3 threadsPerBlock(client_num_of_periods);

    // TODO scan small <<<numBlocks, threadsPerBlock>>>
    int *sums;
    cudaMalloc((void **)&sums, sizeof(sums[0]));
    scan<<<numBlocks, threadsPerBlock>>>(account, changes, periods, sums);
  }
}

__global__ void copy_first_column(int *output, int *input, int clients,
                                  int periods) {
  int threadID = threadIdx.x;

  output[threadID] = input[threadID * clients];
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {
  // copy first collumn of changes to a new array, all is on gpu
  int *n;
  cudaMalloc((void **)&n, periods * sizeof(n[0]));
  copy_first_column<<<1, periods>>>(n, changes, clients, periods);
  printf("n: ");
  cp_from_device_and_print(n, periods);

  int *out;
  cudaMalloc((void **)&out, periods * sizeof(out[0]));
  cudaMemset(out, 0, periods * sizeof(out[0]));

  int *sums;
  cudaMalloc((void **)&sums, sizeof(sums[0]));
  cudaMemset(out, 0, sizeof(out[0]));

  cp_from_device_and_print(out, periods);

  int shared_mem_size = periods * sizeof(int);
  scan<<<1, periods / 2, shared_mem_size>>>(out, n, periods, sums);
  cp_from_device_and_print(out, periods);
  printf("sums: ");
  cp_from_device_and_print(sums, 1);
}
