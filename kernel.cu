#define BLOCK_SIZE 128
#define ROWS_PER_ITER 8

__global__ void kernel(int *twod, int *prefix_sum_per_col, int *reduce_per_row,
                       int cols, int rows) {

  int clientId = blockIdx.x * blockDim.x + threadIdx.x;
  int rows_entries_of_iter[ROWS_PER_ITER]; // array to store rows entries

  int previous_sum = 0;

  // always load/store multiple entries at once
  for (int iter = 0; iter < rows / ROWS_PER_ITER; iter++) {

    // load multiple entries from global memory
    for (int row_in_iter = 0; row_in_iter < ROWS_PER_ITER; row_in_iter++) {
      rows_entries_of_iter[row_in_iter] =
          twod[(iter * ROWS_PER_ITER + row_in_iter) * cols + clientId];
      /**
       * (iter => number of iteration
       * * ROWS_PER_ITER => number of rows to load/store at once
       * + row_in_iter => row index in the current iteration
       * ) * cols => offsets to the row as we need to move by rowIndex * cols to
       * the correct row
       * + clientId => column index, as we need to move by colIndex to the
       * correct column
       */
    }

    // // Debug print before modification
    // printf("Client %d | Iter: %d | Before: ", clientId, iter);
    // for (int i = 0; i < ROWS_PER_ITER; i++) {
    //   printf("%d ", rows_entries_of_iter[i]);
    // }
    // printf("\n");

    // calculate prefix sum for each row
    rows_entries_of_iter[0] += previous_sum;
    for (int i = 1; i < ROWS_PER_ITER; i++) {
      rows_entries_of_iter[i] += rows_entries_of_iter[i - 1];
    }

    // // Debug print after modification
    // printf("Client %d | Iter: %d | After: ", clientId, iter);
    // for (int i = 0; i < ROWS_PER_ITER; i++) {
    //   printf("%d ", rows_entries_of_iter[i]);
    // }
    // printf("\n");

    // store multiple entries to global memory
    for (int row_in_iter = 0; row_in_iter < ROWS_PER_ITER; row_in_iter++) {
      prefix_sum_per_col[(iter * ROWS_PER_ITER + row_in_iter) * cols +
                         clientId] = rows_entries_of_iter[row_in_iter];
    }

    // update previous sum
    previous_sum = rows_entries_of_iter[ROWS_PER_ITER - 1];
  }
}

/**
 * @brief Calculates account state and sum of each period based on the changes
 * array. The changes array is a semi 2d array where each row represents a
 * period and each column represents a client.
 *
 * Example: client = 3, period = 2
 * => changes[2 * clients + 3]
 *
 * All the arrays mentioned are already allocated on the device.
 *
 * @param changes The changes array.
 * @param account The account array.
 * @param sum The sum array.
 * @param clients The number of clients.
 * @param periods The number of periods.
 */
void solveGPU(int *changes, int *account, int *sum, int clients, int periods) {

  kernel<<<clients / BLOCK_SIZE, BLOCK_SIZE>>>(changes, account, sum, clients,
                                               periods);

  // print cuda error if any
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
}
