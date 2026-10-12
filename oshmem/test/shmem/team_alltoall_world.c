/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* Team alltoall on SHMEM_TEAM_WORLD (odd PE count): int, long, uint64, double.
 *
 * NP: 3
 */

#include <stdint.h>
#include <stdio.h>
#include <shmem.h>

#define MAXN   8   /* largest team handled */
#define MAXNEL 3   /* largest nelems used */
#define MAXSTR 3   /* largest stride used */

static int failures = 0;
static int my_world_pe = -1;

#define FAIL(label, fmt, ...)                                            \
    do {                                                                 \
        if (failures++ < 20) {                                           \
            printf("FAIL: PE %d: %s: " fmt "\n", my_world_pe, label,    \
                   __VA_ARGS__);                                         \
        }                                                                \
    } while (0)


/* Value sent by team PE "from" to team PE "to", element e of its block */
#define A_VAL(TYPE, from, to, e, round) \
    ((TYPE)((round) * 100000 + ((from) + 1) * 1000 + (to) * 10 + (e)))

#define DEFINE_ALLTOALL(NAME, TYPE, FN)                                      \
static void test_##NAME(shmem_team_t team, const char *label)                \
{                                                                            \
    static TYPE src[2][MAXN * MAXNEL];                                       \
    static TYPE dst[2][MAXN * MAXNEL];                                       \
    int me = shmem_team_my_pe(team);                                         \
    int n = shmem_team_n_pes(team);                                          \
    int nels[2] = { 1, MAXNEL };                                             \
    int k, r, i, e, j, rc;                                                   \
    char l[128];                                                             \
                                                                             \
    for (k = 0; k < 2; k++) {                                                \
        int nel = nels[k];                                                   \
        shmem_team_sync(team);                                               \
        for (r = 0; r < 2; r++) {                                            \
            for (j = 0; j < n; j++) {                                        \
                for (e = 0; e < nel; e++) {                                  \
                    src[r][j * nel + e] = A_VAL(TYPE, me, j, e, r + 1);      \
                }                                                            \
            }                                                                \
            for (j = 0; j < MAXN * MAXNEL; j++) {                            \
                dst[r][j] = (TYPE)-1;                                        \
            }                                                                \
        }                                                                    \
        shmem_team_sync(team);                                               \
        /* back-to-back calls with distinct buffers */                       \
        for (r = 0; r < 2; r++) {                                            \
            rc = FN(team, dst[r], src[r], (size_t)nel);                      \
            if (0 != rc) {                                                   \
                FAIL(label, #FN " nelems %d round %d returned %d, "          \
                     "expected 0", nel, r, rc);                              \
            }                                                                \
        }                                                                    \
        for (r = 0; r < 2; r++) {                                            \
            for (i = 0; i < n; i++) {                                        \
                for (e = 0; e < nel; e++) {                                  \
                    TYPE exp = A_VAL(TYPE, i, me, e, r + 1);                 \
                    TYPE got = dst[r][i * nel + e];                          \
                    if (got != exp) {                                        \
                        snprintf(l, sizeof(l), "%s n=%d", label, n);         \
                        FAIL(l, #FN " nelems %d call %d block %d elem %d: " \
                             "expected %.0f actual %.0f", nel, r, i, e,      \
                             (double)exp, (double)got);                      \
                    }                                                        \
                }                                                            \
            }                                                                \
            /* nothing beyond n * nel elements may be touched */             \
            for (j = n * nel; j < MAXN * MAXNEL; j++) {                      \
                if (dst[r][j] != (TYPE)-1) {                                 \
                    FAIL(label, #FN " wrote past end of dest at %d", j);     \
                }                                                            \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

DEFINE_ALLTOALL(int, int, shmem_int_alltoall)
DEFINE_ALLTOALL(long, long, shmem_long_alltoall)
DEFINE_ALLTOALL(uint64, uint64_t, shmem_uint64_alltoall)
DEFINE_ALLTOALL(double, double, shmem_double_alltoall)

static void test_all_types(shmem_team_t team, const char *label)
{
    test_int(team, label);
    test_long(team, label);
    test_uint64(team, label);
    test_double(team, label);
}

int main(void)
{
    shmem_init();
    my_world_pe = shmem_my_pe();
    if (shmem_n_pes() < 3) {
        if (0 == my_world_pe) {
            printf("SKIP: need at least 3 PEs\n");
        }
        shmem_finalize();
        return 77;
    }

    test_all_types(SHMEM_TEAM_WORLD, "world team");

    shmem_barrier_all();
    shmem_finalize();
    return failures ? 1 : 0;
}
