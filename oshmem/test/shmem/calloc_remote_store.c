/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * NP: 4
 *
 * shmem_calloc must finish zeroing its block before the barrier that ends
 * the allocation.  Otherwise the PE that arrives last at the barrier leaves
 * it first and can store into the blocks of the other PEs before they have
 * zeroed them, and the late zeroing erases the store.
 *
 * In each iteration one PE (rotating) is delayed before calling
 * shmem_calloc.  As soon as it returns, it stores a nonzero value into the
 * last word of the block on every other PE; the last word is the one that
 * memset reaches last, which widens the window.  Every other PE then waits
 * for that store with a bounded poll, so a lost store is reported as a
 * failure instead of a hang.
 */

#include <stdio.h>
#include <unistd.h>
#include <shmem.h>

#define ITERATIONS 60
#define BLOCK_INTS (1024 * 1024)
#define DELAY_USEC 20000
#define POLL_LIMIT 2000000

int main(void)
{
    int me, npes, it, i;
    int failures = 0;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 2) {
        shmem_finalize();
        return 77;
    }

    for (it = 0; it < ITERATIONS; ++it) {
        int late = it % npes;
        int expected = it + 1;
        int *block;
        long polls = 0;

        if (me == late) {
            usleep(DELAY_USEC);
        }

        block = (int *) shmem_calloc(BLOCK_INTS, sizeof(int));
        if (NULL == block) {
            fprintf(stderr, "FAIL: PE %d: shmem_calloc returned NULL\n", me);
            ++failures;
            continue;
        }

        if (me == late) {
            for (i = 0; i < npes; ++i) {
                if (i != me) {
                    shmem_int_p(&block[BLOCK_INTS - 1], expected, i);
                }
            }
            shmem_quiet();
        } else {
            while (expected != ((volatile int *) block)[BLOCK_INTS - 1]
                   && polls < POLL_LIMIT) {
                ++polls;
            }
            if (expected != block[BLOCK_INTS - 1]) {
                fprintf(stderr, "FAIL: PE %d: iteration %d: store from PE %d "
                        "was lost (got %d, expected %d)\n",
                        me, it, late, block[BLOCK_INTS - 1], expected);
                ++failures;
            }
        }

        shmem_free(block);
    }

    shmem_finalize();
    return failures ? 1 : 0;
}
