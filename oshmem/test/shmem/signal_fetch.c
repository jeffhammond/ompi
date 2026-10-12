/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* NP: 3
 *
 * Cross-check shmem_signal_fetch against shmem_signal_wait_until: after a
 * wait returns, the fetched value is at least the returned value, and once
 * all updates are complete both agree with the expected total.
 */

#include <stdio.h>
#include <stdint.h>
#include <shmem.h>

static uint64_t sig;
static int data;

int main(void)
{
    int me, npes, left, right, fails = 0;
    uint64_t got, fetched;
    int val;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2) {
        shmem_finalize();
        return 77;
    }
    left = (me + npes - 1) % npes;
    right = (me + 1) % npes;

    sig = 0;
    data = -1;
    shmem_barrier_all();

    fetched = shmem_signal_fetch(&sig);
    if (0 != fetched) {
        printf("FAIL: PE %d: initial fetch %llu, expected 0\n", me,
               (unsigned long long) fetched);
        ++fails;
    }
    shmem_barrier_all();

    val = 100 + me;
    shmem_int_put_signal(&data, &val, 1, &sig, 5, SHMEM_SIGNAL_ADD, right);
    got = shmem_signal_wait_until(&sig, SHMEM_CMP_GE, 5);
    fetched = shmem_signal_fetch(&sig);
    if (5 != got || 5 != fetched) {
        printf("FAIL: PE %d: wait returned %llu fetch %llu, expected 5\n",
               me, (unsigned long long) got, (unsigned long long) fetched);
        ++fails;
    }
    if (data != 100 + left) {
        printf("FAIL: PE %d: data %d, expected %d\n", me, data, 100 + left);
        ++fails;
    }
    shmem_barrier_all();

    /* second update on top of the first */
    val = 200 + me;
    shmem_int_put_signal(&data, &val, 1, &sig, 6, SHMEM_SIGNAL_ADD, right);
    got = shmem_signal_wait_until(&sig, SHMEM_CMP_EQ, 11);
    fetched = shmem_signal_fetch(&sig);
    if (11 != got || 11 != fetched) {
        printf("FAIL: PE %d: wait returned %llu fetch %llu, expected 11\n",
               me, (unsigned long long) got, (unsigned long long) fetched);
        ++fails;
    }
    if (data != 200 + left) {
        printf("FAIL: PE %d: data %d, expected %d\n", me, data, 200 + left);
        ++fails;
    }

    shmem_finalize();
    return fails ? 1 : 0;
}
