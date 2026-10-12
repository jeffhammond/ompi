/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* NP: 4
 *
 * Several senders concurrently add distinct bit values to the signal on
 * PE 0 while it waits.  The value returned by shmem_signal_wait_until must
 * be a value the signal actually held (a partial sum of the senders'
 * contributions) and must satisfy the condition, not a later value.
 */

#include <stdio.h>
#include <stdint.h>
#include <shmem.h>

#define ROUNDS 20

static uint64_t sig;
static int flag[8];

int main(void)
{
    int me, npes, r, i, fails = 0;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 3) {
        shmem_finalize();
        return 77;
    }

    for (r = 0; r < ROUNDS; ++r) {
        uint64_t full = 0, got, fetched;
        int spin;

        for (i = 1; i < npes; ++i) {
            full += (uint64_t) 1 << i;
        }
        sig = 0;
        shmem_barrier_all();

        if (0 == me) {
            /* NE 0: first change; must be a nonzero subset of the bits */
            got = shmem_signal_wait_until(&sig, SHMEM_CMP_NE, 0);
            if (0 == got || (got & ~full) != 0) {
                printf("FAIL: PE 0 round %d: NE returned 0x%llx, "
                       "not a subset of 0x%llx\n", r,
                       (unsigned long long) got, (unsigned long long) full);
                ++fails;
            }
            /* GT: every contribution is >= 2, so any nonzero value qualifies */
            got = shmem_signal_wait_until(&sig, SHMEM_CMP_GT, 1);
            if (got <= 1 || (got & ~full) != 0) {
                printf("FAIL: PE 0 round %d: GT returned 0x%llx, "
                       "expected > 0x%llx\n", r, (unsigned long long) got,
                       (unsigned long long) 1);
                ++fails;
            }
            /* signal only grows, so a later fetch is >= returned value */
            fetched = shmem_signal_fetch(&sig);
            if (fetched < got) {
                printf("FAIL: PE 0 round %d: fetch 0x%llx < returned 0x%llx\n",
                       r, (unsigned long long) fetched,
                       (unsigned long long) got);
                ++fails;
            }
            got = shmem_signal_wait_until(&sig, SHMEM_CMP_EQ, full);
            if (got != full) {
                printf("FAIL: PE 0 round %d: EQ returned 0x%llx, "
                       "expected 0x%llx\n", r, (unsigned long long) got,
                       (unsigned long long) full);
                ++fails;
            }
        } else {
            int one = me;
            /* stagger senders so the signal changes while PE 0 waits */
            for (spin = 0; spin < me * 2000 * (r % 3 + 1); ++spin) {
                flag[0] = spin;
            }
            shmem_int_put_signal(&flag[me], &one, 1, &sig,
                                 (uint64_t) 1 << me, SHMEM_SIGNAL_ADD, 0);
        }
        shmem_barrier_all();
        if (0 == me) {
            for (i = 1; i < npes; ++i) {
                if (flag[i] != i) {
                    printf("FAIL: PE 0 round %d: flag[%d] = %d, expected %d\n",
                           r, i, flag[i], i);
                    ++fails;
                }
                flag[i] = 0;
            }
        }
        shmem_barrier_all();
    }

    shmem_finalize();
    return fails ? 1 : 0;
}
