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
#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK THREADS_PER_BLOCK * 2

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
__global__ void prescan_arbitrary(int *output, int *input, int n,
                                  int powerOfTwo) {
  extern __shared__ int temp[]; // allocated on invocation
  int threadID = threadIdx.x;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

  if (threadID < n) {
    temp[ai + bankOffsetA] = input[ai];
    temp[bi + bankOffsetB] = input[bi];
  } else {
    temp[ai + bankOffsetA] = 0;
    temp[bi + bankOffsetB] = 0;
  }

  int offset = 1;
  for (int d = powerOfTwo >> 1; d > 0;
       d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (threadID == 0) {
    temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] =
        0; // clear the last element
  }

  for (int d = 1; d < powerOfTwo; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  if (threadID < n) {
    output[ai] = temp[ai + bankOffsetA];
    output[bi] = temp[bi + bankOffsetB];
  }
}

__global__ void prescan_large(int *output, int *input, int n, int *sums) {
  extern __shared__ int temp[];

  int blockID = blockIdx.x;
  int threadID = threadIdx.x;
  int blockOffset = blockID * n;

  int ai = threadID;
  int bi = threadID + (n / 2);
  int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
  int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
  temp[ai + bankOffsetA] = input[blockOffset + ai];
  temp[bi + bankOffsetB] = input[blockOffset + bi];

  int offset = 1;
  for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
  {
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  __syncthreads();

  if (threadID == 0) {
    sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
    temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
  }

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();
    if (threadID < d) {
      int ai = offset * (2 * threadID + 1) - 1;
      int bi = offset * (2 * threadID + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  output[blockOffset + ai] = temp[ai + bankOffsetA];
  output[blockOffset + bi] = temp[bi + bankOffsetB];
}
/*///////////////////////////////////*/

/*///////////scan funcs//////////////*/
void scanSmallDeviceArray(int *account, int *changes, int length) {
  prescan_arbitrary<<<1, (length + 1) / 2>>>(account, changes, length, length);
}

void scanLarge(int *account, int *changes, int length) {
  const int blocks = length / ELEMENTS_PER_BLOCK;
  const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

  int *d_sums, *d_incr;
  cudaMalloc((void **)&d_sums, blocks * sizeof(int));
  cudaMalloc((void **)&d_incr, blocks * sizeof(int));

  prescan_large<<<blocks, THREADS_PER_BLOCK, 2 * sharedMemArraySize>>>(
      account, changes, ELEMENTS_PER_BLOCK, d_sums);

  const int sumsArrThreadsNeeded = (blocks + 1) / 2;
  if (sumsArrThreadsNeeded > THREADS_PER_BLOCK) {
    // perform a large scan on the sums arr
    // scanLargeDeviceArray(d_incr, d_sums, blocks);
    scanLargeEvenDeviceArray(d_incr, d_sums, blocks);
  } else {
    // only need one block to scan sums arr so can use small scan
    scanSmallDeviceArray(d_incr, d_sums, blocks);
  }

  add<<<blocks, ELEMENTS_PER_BLOCK>>>(account, ELEMENTS_PER_BLOCK, d_incr);

  cudaFree(d_sums);
  cudaFree(d_incr);
}
/*///////////////////////////////////*/

void calc_accounts(int *account, int *changes, int clients, int periods) {
  int client_num_of_periods = periods;
  if (client_num_of_periods > ELEMENTS_PER_BLOCK) {
    // perform a large scan (meaning more than one block per client)
    // to distinguish the clients blocks from each other we define 2Dim grid
    // where the x axis is the client and the y axis is the block
    int remainder = client_num_of_periods % ELEMENTS_PER_BLOCK;
    // if we can fit the clients periods exactly in the blocks with max elements
    // per block
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
  }
}

void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {

  calc_accounts(account, changes, clients, periods);
}
