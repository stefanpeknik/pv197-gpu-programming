def calculate_distribution(X, Y):
    # Constants
    MAX_THREADS_PER_BLOCK = 1024
    PERIODS_PER_THREAD = 2
    PERIODS_PER_BLOCK = MAX_THREADS_PER_BLOCK * PERIODS_PER_THREAD

    # Calculate how many blocks are required for each client's periods
    blocks_per_client = (Y + PERIODS_PER_BLOCK - 1) // PERIODS_PER_BLOCK

    # Calculate the number of threads needed per block for periods of each client
    # Each block processes at most PERIODS_PER_BLOCK periods
    periods_handled_per_block = min(Y, PERIODS_PER_BLOCK)
    threads_per_block = periods_handled_per_block // PERIODS_PER_THREAD

    # Calculate how many clients can fit in one block if their periods do not exceed the block's limit
    if Y <= PERIODS_PER_BLOCK:
        clients_per_block = PERIODS_PER_BLOCK // Y
        total_blocks = (X + clients_per_block - 1) // clients_per_block
    else:
        # If each client requires multiple blocks, total blocks scale with clients
        clients_per_block = 1
        total_blocks = X * blocks_per_client

    print(f"Each client requires {blocks_per_client} block(s) to cover their periods.")
    print(
        f"One block can handle {clients_per_block} client(s) with {Y} periods each (if Y <= {PERIODS_PER_BLOCK})."
    )
    print(f"Threads needed per block: {threads_per_block}")
    print(f"Total blocks needed: {total_blocks}")

    # Output values
    return {
        "blocks_per_client": blocks_per_client,
        "clients_per_block": clients_per_block,
        "threads_per_block": threads_per_block,
        "total_blocks": total_blocks,
    }


# Example usage
X = 128  # Number of clients
Y = 8192  # Number of periods per client
result = calculate_distribution(X, Y)

# Display results
print("Input:")
print(f"Number of clients: {X}")
print(f"Number of periods per client: {Y}")
print("\nResults:")
print(f"Blocks per client: {result['blocks_per_client']}")
print(f"Clients per block: {result['clients_per_block']}")
print(f"Threads per block: {result['threads_per_block']}")


# clients = 128 * 3
# periods = 128 * 3  # bassically elements per client
# threads = 1024
# elements_per_thread = lambda: threads * 2

# clients_per_block = clients

# while clients_per_block * periods > elements_per_thread():
#     print(
#         f"Clients per block: {clients_per_block}, Total elements: {clients_per_block * periods}, Elements per thread: {elements_per_thread()}"
#     )
#     clients_per_block //= 2

# print(
#     f"Clients per block: {clients_per_block}, Total elements: {clients_per_block * periods}, Elements per thread: {elements_per_thread()}"
# )

# print("-------")

# threads_per_client = periods
