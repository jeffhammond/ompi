/*
 * Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/* Team alltoalls on SHMEM_TEAM_WORLD (odd PE count) with dst/sst strides > 1.
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

static const ptrdiff_t strides[3][2] = { {1, 1}, {2, 3}, {3, 2} }; /* dst, sst */

#define DEFINE_ALLTOALLS(NAME, TYPE, FN)                                     \
static void test_##NAME(shmem_team_t team, const char *label)                \
{                                                                            \
    static TYPE src[2][MAXN * MAXNEL * MAXSTR];                              \
    static TYPE dst[2][MAXN * MAXNEL * MAXSTR];                              \
    int me = shmem_team_my_pe(team);                                         \
    int n = shmem_team_n_pes(team);                                          \
    int nels[3] = { 1, 2, MAXNEL };                                          \
    int k, s, r, i, e, j, rc;                                                \
    char l[128];                                                             \
                                                                             \
    for (s = 0; s < 3; s++) {                                                \
        ptrdiff_t dstr = strides[s][0], sstr = strides[s][1];                \
        int nel = nels[s];                                                   \
        int len = n * nel;                                                   \
        for (k = 0; k < 2; k++) {                                            \
            shmem_team_sync(team);                                           \
            for (r = 0; r < 2; r++) {                                        \
                for (j = 0; j < MAXN * MAXNEL * MAXSTR; j++) {               \
                    src[r][j] = (TYPE)-2; /* gap marker */                   \
                    dst[r][j] = (TYPE)-1; /* sentinel */                     \
                }                                                            \
                for (j = 0; j < n; j++) {                                    \
                    for (e = 0; e < nel; e++) {                              \
                        src[r][(j * nel + e) * sstr] =                       \
                            A_VAL(TYPE, me, j, e, r + 1 + k);                \
                    }                                                        \
                }                                                            \
            }                                                                \
            shmem_team_sync(team);                                           \
            for (r = 0; r < 2; r++) {                                        \
                rc = FN(team, dst[r], src[r], dstr, sstr, (size_t)nel);      \
                if (0 != rc) {                                               \
                    FAIL(label, #FN " dst %d sst %d round %d returned %d, "  \
                         "expected 0", (int)dstr, (int)sstr, r, rc);         \
                }                                                            \
            }                                                                \
            for (r = 0; r < 2; r++) {                                        \
                for (j = 0; j < MAXN * MAXNEL * MAXSTR; j++) {               \
                    TYPE exp = (TYPE)-1;                                     \
                    TYPE got = dst[r][j];                                    \
                    if (0 == j % dstr && j / dstr < len) {                   \
                        i = (j / dstr) / nel;                                \
                        e = (j / dstr) % nel;                                \
                        exp = A_VAL(TYPE, i, me, e, r + 1 + k);              \
                    }                                                        \
                    if (got != exp) {                                        \
                        snprintf(l, sizeof(l), "%s n=%d", label, n);         \
                        FAIL(l, #FN " dst %d sst %d nelems %d call %d "     \
                             "index %d: expected %.0f actual %.0f",          \
                             (int)dstr, (int)sstr, nel, r, j,                \
                             (double)exp, (double)got);                      \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
    }                                                                        \
}

DEFINE_ALLTOALLS(int, int, shmem_int_alltoalls)
DEFINE_ALLTOALLS(long, long, shmem_long_alltoalls)
DEFINE_ALLTOALLS(uint64, uint64_t, shmem_uint64_alltoalls)
DEFINE_ALLTOALLS(double, double, shmem_double_alltoalls)

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
