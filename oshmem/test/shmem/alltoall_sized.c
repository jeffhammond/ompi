/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* Active-set shmem_alltoall32/64 and shmem_alltoalls32/64, including strides
 * greater than 1 and active sets that are not the whole job.
 *
 * NP: 4
 */

#include <stdint.h>
#include <stdio.h>
#include <shmem.h>

#define MAXN   4
#define MAXNEL 3
#define MAXSTR 3

static long psync[SHMEM_ALLTOALL_SYNC_SIZE];
static long psyncs[SHMEM_ALLTOALLS_SYNC_SIZE];
static int failures = 0;
static int me = -1;

#define VAL(from, to, e, round) \
    ((uint64_t)((round) * 100000 + ((from) + 1) * 1000 + (to) * 10 + (e)))

#define DEFINE_TEST(NAME, TYPE, CALL, STRIDED)                               \
static void NAME(int start, int logstride, int size, int round)              \
{                                                                            \
    static TYPE src[MAXN * MAXNEL * MAXSTR];                                 \
    static TYPE dst[MAXN * MAXNEL * MAXSTR];                                 \
    ptrdiff_t dstr = (STRIDED) ? 2 : 1, sstr = (STRIDED) ? 3 : 1;            \
    int nel = MAXNEL;                                                        \
    int step = 1 << logstride;                                               \
    int member = (me >= start && 0 == (me - start) % step &&                 \
                  (me - start) / step < size);                               \
    int idx = member ? (me - start) / step : -1;                             \
    int i, e, j;                                                             \
                                                                             \
    for (j = 0; j < MAXN * MAXNEL * MAXSTR; j++) {                           \
        src[j] = (TYPE)-2;                                                   \
        dst[j] = (TYPE)-1;                                                   \
    }                                                                        \
    for (j = 0; member && j < size; j++) {                                   \
        for (e = 0; e < nel; e++) {                                          \
            src[(j * nel + e) * sstr] = (TYPE)VAL(idx, j, e, round);         \
        }                                                                    \
    }                                                                        \
    shmem_barrier_all();                                                     \
    if (member) {                                                            \
        CALL;                                                                \
    }                                                                        \
    shmem_barrier_all();                                                     \
    for (j = 0; j < MAXN * MAXNEL * MAXSTR; j++) {                           \
        TYPE exp = (TYPE)-1;                                                 \
        if (member && 0 == j % dstr && j / dstr < size * nel) {              \
            i = (j / dstr) / nel;                                            \
            e = (j / dstr) % nel;                                            \
            exp = (TYPE)VAL(i, idx, e, round);                               \
        }                                                                    \
        if (dst[j] != exp) {                                                 \
            if (failures++ < 20) {                                           \
                printf("FAIL: PE %d: " #NAME " set (%d,%d,%d) index %d: "    \
                       "expected %llu actual %llu\n", me, start,             \
                       logstride, size, j, (unsigned long long)exp,          \
                       (unsigned long long)dst[j]);                          \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

DEFINE_TEST(test_alltoall32, uint32_t,
            shmem_alltoall32(dst, src, nel, start, logstride, size, psync),
            0)
DEFINE_TEST(test_alltoall64, uint64_t,
            shmem_alltoall64(dst, src, nel, start, logstride, size, psync),
            0)
DEFINE_TEST(test_alltoalls32, uint32_t,
            shmem_alltoalls32(dst, src, dstr, sstr, nel, start, logstride,
                              size, psyncs),
            1)
DEFINE_TEST(test_alltoalls64, uint64_t,
            shmem_alltoalls64(dst, src, dstr, sstr, nel, start, logstride,
                              size, psyncs),
            1)

int main(void)
{
    static const int sets[3][3] = { {0, 0, 4}, {1, 0, 3}, {0, 1, 2} };
    int s, round, i;

    shmem_init();
    me = shmem_my_pe();
    if (shmem_n_pes() < 4) {
        if (0 == me) {
            printf("SKIP: need at least 4 PEs\n");
        }
        shmem_finalize();
        return 77;
    }

    for (i = 0; i < SHMEM_ALLTOALL_SYNC_SIZE; i++) {
        psync[i] = SHMEM_SYNC_VALUE;
    }
    for (i = 0; i < SHMEM_ALLTOALLS_SYNC_SIZE; i++) {
        psyncs[i] = SHMEM_SYNC_VALUE;
    }
    shmem_barrier_all();

    for (s = 0; s < 3; s++) {
        for (round = 1; round <= 2; round++) {
            test_alltoall32(sets[s][0], sets[s][1], sets[s][2], round);
            test_alltoall64(sets[s][0], sets[s][1], sets[s][2], round);
            test_alltoalls32(sets[s][0], sets[s][1], sets[s][2], round);
            test_alltoalls64(sets[s][0], sets[s][1], sets[s][2], round);
        }
    }

    shmem_barrier_all();
    shmem_finalize();
    return failures ? 1 : 0;
}
