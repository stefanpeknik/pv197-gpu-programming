#include <iostream>

void print_array(int *array, int size) {
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

void print_clients(int *two_d, int clients, int periods) {
  for (int c = 0; c < clients; c++) {
    printf("Client %d: ", c);
    for (int p = 0; p < periods; p++) {
      printf("%d ", two_d[p * clients + c]);
    }
    printf("\n");
  }
}

void print_sum(int *sum, int periods) {
  for (int i = 0; i < periods; i++) {
    printf("%d ", sum[i]);
  }
  printf("\n");
}

void solveCPU(int *changes, int *account, int *sum, int clients, int periods) {
  // printf("changes\n");
  // print_clients(changes, clients, periods);

  for (int i = 0; i < clients; i++)
    account[i] = changes[i]; // the first change is copied

  for (int j = 1; j < periods; j++) {
    for (int i = 0; i < clients; i++) {
      account[j * clients + i] =
          account[(j - 1) * clients + i] + changes[j * clients + i];
    }
  }

  // printf("account\n");
  // print_clients(account, clients, periods);

  for (int j = 0; j < periods; j++) {
    int s = 0;
    for (int i = 0; i < clients; i++) {
      s += account[j * clients + i];
    }
    sum[j] = s;
  }

  printf("sum\n");
  print_sum(sum, periods);
}
