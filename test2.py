def calculate_even_client_distribution(X):
    # Constants
    MAX_THREADS_PER_BLOCK = 1024

    # Determine the largest even divisor of X that is <= MAX_THREADS_PER_BLOCK
    for clients_per_block in range(MAX_THREADS_PER_BLOCK, 0, -2):
        print(f"Trying clients per block: {clients_per_block}")
        if X % clients_per_block == 0:
            break

    # Calculate the number of blocks needed
    blocks_needed = X // clients_per_block

    print(f"Total blocks needed: {blocks_needed}")
    print(f"Clients per block: {clients_per_block}")
    print(
        f"Threads per block: {clients_per_block} (since each thread handles one client)"
    )

    # Output values
    return {
        "total_blocks": blocks_needed,
        "clients_per_block": clients_per_block,
        "threads_per_block": clients_per_block,  # Same as clients per block in this setup
    }


# Example usage
X = 128 * 30  # Example number of clients
clients_distribution = calculate_even_client_distribution(X)

# Display results
print("Input:")
print(f"Number of clients: {X}")
print("\nResults:")
print(f"Total blocks required: {clients_distribution['total_blocks']}")
print(f"Clients per block: {clients_distribution['clients_per_block']}")
print(f"Threads per block: {clients_distribution['threads_per_block']}")
