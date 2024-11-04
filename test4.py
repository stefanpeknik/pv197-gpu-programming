cols = 4
rows = 8

blocks_per_col = 2
threads_per_block = 4

arr = []
for i in range(rows * cols):
    arr.append(i)
print(arr)

for i in range(rows):
    for j in range(cols):
        print(f"{arr[i * cols + j]:3}", end=" ")
    print()
print()


def get_index(blockId, threadId):
    col = blockId // blocks_per_col  # Determine the column
    starting_row = (
        blockId % blocks_per_col
    ) * threads_per_block  # Row where this block starts in the column
    return (starting_row + threadId) * cols + col


for block in range(blocks_per_col * cols):
    for thread in range(threads_per_block):
        # print block, thread, index from find_val_in_arr, val on that index
        print(
            f"block: {block}, thread: {thread} \
            index: {get_index(block, thread)}, val: {arr[get_index(block, thread)]}"
        )
