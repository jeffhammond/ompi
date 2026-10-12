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
 * Ring of PEs: each PE put-signals (SHMEM_SIGNAL_ADD) to its right
 * neighbor and waits on its own signal with every comparison operator.
 * The returned value must satisfy the comparison and the data put before
 * the signal must be visible once the wait returns.
 */

#include <stdio.h>
#include <stdint.h>
#include <shmem.h>

#define NDATA 64

static uint64_t sig;
static int data[NDATA];
static int src[NDATA];

static int check_cmp(int cmp, uint64_t v, uint64_t arg)
{
    switch (cmp) {
    case SHMEM_CMP_EQ: return v == arg;
    case SHMEM_CMP_NE: return v != arg;
    case SHMEM_CMP_GT: return v > arg;
    case SHMEM_CMP_GE: return v >= arg;
    case SHMEM_CMP_LT: return v < arg;
    case SHMEM_CMP_LE: return v <= arg;
    }
    return 0;
}

int main(void)
{
    static const struct {
        const char *name;
        int cmp;
        uint64_t init;
        uint64_t add;       /* added by the left neighbor */
        uint64_t arg;
        uint64_t expect;    /* value after the add */
    } cases[] = {
        { "EQ", SHMEM_CMP_EQ, 0, 7, 7, 7 },
        { "GE", SHMEM_CMP_GE, 0, 3, 3, 3 },
        { "GT", SHMEM_CMP_GT, 0, 4, 3, 4 },
        { "NE", SHMEM_CMP_NE, 0, 5, 0, 5 },
        /* decrement by adding 2^64-1 */
        { "LE", SHMEM_CMP_LE, 5, UINT64_MAX, 4, 4 },
        { "LT", SHMEM_CMP_LT, 5, UINT64_MAX, 5, 4 },
    };
    int me, npes, left, right, c, i, fails = 0;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2) {
        shmem_finalize();
        return 77;
    }
    left = (me + npes - 1) % npes;
    right = (me + 1) % npes;

    for (c = 0; c < (int) (sizeof(cases) / sizeof(cases[0])); ++c) {
        uint64_t got;

        sig = cases[c].init;
        for (i = 0; i < NDATA; ++i) {
            data[i] = -1;
            src[i] = me * 1000 + c * 100 + i;
        }
        shmem_barrier_all();

        shmem_int_put_signal(data, src, NDATA, &sig, cases[c].add,
                             SHMEM_SIGNAL_ADD, right);
        got = shmem_signal_wait_until(&sig, cases[c].cmp, cases[c].arg);

        if (!check_cmp(cases[c].cmp, got, cases[c].arg)
            || got != cases[c].expect) {
            printf("FAIL: PE %d cmp %s: returned %llu, expected %llu\n", me,
                   cases[c].name, (unsigned long long) got,
                   (unsigned long long) cases[c].expect);
            ++fails;
        }
        for (i = 0; i < NDATA; ++i) {
            int exp = left * 1000 + c * 100 + i;
            if (data[i] != exp) {
                printf("FAIL: PE %d cmp %s: data[%d] = %d, expected %d\n",
                       me, cases[c].name, i, data[i], exp);
                ++fails;
                break;
            }
        }
        shmem_barrier_all();
    }

    shmem_finalize();
    return fails ? 1 : 0;
}
