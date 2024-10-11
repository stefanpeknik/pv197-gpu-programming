void solveCPU(int *changes, int *account, int *sum, int clients, int periods) {
    for (int i = 0; i < clients; i++)
        account[i] = changes[i]; // the first change is copied
    for (int j = 1; j < periods; j++) {
        for (int i = 0; i < clients; i++) {
            account[j*clients + i] = account[(j-1)*clients + i] 
                + changes[j*clients + i];
        }
    }

    for (int j = 0; j < periods; j++) {
        int s = 0;
        for (int i = 0; i < clients; i++) {
            s += account[j*clients + i];
        }
        sum[j] = s;
    }
}

