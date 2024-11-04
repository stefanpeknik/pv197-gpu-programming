#   int threads_per_client = periods / 2;
#   int threads_per_block_per_client;
#   if (threads_per_client <= MAX_THREADS_PER_BLOCK)
#     threads_per_block_per_client = threads_per_client;
#   } else {
#     threads_per_block_per_client = MAX_THREADS_PER_BLOCK;
#     while (threads_per_client % threads_per_block_per_client != 0) {
#       threads_per_block_per_client /= 2;
#     }
#   }
#   int blocks_per_client = threads_per_client / threads_per_block_per_client;
#   int blocks = clients * blocks_per_client;

clients = 128 * 12
periods = 128 * 20
MAX_THREADS_PER_BLOCK = 1024

threads_per_client = periods / 2
threads_per_block_per_client = 0
if threads_per_client <= MAX_THREADS_PER_BLOCK:
    threads_per_block_per_client = threads_per_client
else:
    threads_per_block_per_client = MAX_THREADS_PER_BLOCK
    while threads_per_client % threads_per_block_per_client != 0:
        threads_per_block_per_client /= 2

blocks_per_client = threads_per_client / threads_per_block_per_client
blocks = clients * blocks_per_client

# print it all
print(f"clients: {clients}")
print(f"periods: {periods}")
print(f"threads_per_client: {threads_per_client}")
print(f"threads_per_block_per_client: {threads_per_block_per_client}")
print(f"blocks_per_client: {blocks_per_client}")
print(f"blocks: {blocks}")
