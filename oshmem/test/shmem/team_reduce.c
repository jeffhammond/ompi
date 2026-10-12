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
 * Team reductions: every operation over a range of types on SHMEM_TEAM_WORLD,
 * strided teams and 2D-split teams, checked against locally computed values,
 * with repeated back-to-back calls on one team (pSync reuse).
 *
 * NP: 8
 */

#include <shmem.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int nfail = 0;

#define FAIL(...)                                          \
    do {                                                   \
        if (nfail < 30) {                                  \
            printf("FAIL: PE %d: ", shmem_my_pe());        \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
            fflush(stdout);                                \
        }                                                  \
        nfail++;                                           \
    } while (0)

/*
 * The team cases every test runs over.  All PEs create the same list in the
 * same order; a PE that is not a member of a case holds SHMEM_TEAM_INVALID.
 */
enum { NCASE = 5 };
static shmem_team_t tm[NCASE];
static const char *tname[NCASE] = { "world", "strided(1,2)", "strided(0,3)",
                                    "2d-x", "2d-y" };

static void make_teams(int n)
{
    int i, rc;
    int xrange = (n > 4) ? 3 : 2;

    for (i = 0; i < NCASE; i++) {
        tm[i] = SHMEM_TEAM_INVALID;
    }
    tm[0] = SHMEM_TEAM_WORLD;
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 1, 2, n / 2, NULL, 0,
                                  &tm[1]);
    if (0 != rc) {
        FAIL("split_strided(1,2) returned %d", rc);
    }
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 3, (n + 2) / 3, NULL, 0,
                                  &tm[2]);
    if (0 != rc) {
        FAIL("split_strided(0,3) returned %d", rc);
    }
    rc = shmem_team_split_2d(SHMEM_TEAM_WORLD, xrange, NULL, 0, &tm[3], NULL, 0,
                             &tm[4]);
    if (0 != rc) {
        FAIL("split_2d(%d) returned %d", xrange, rc);
    }
}

static void destroy_teams(void)
{
    int i;

    for (i = 1; i < NCASE; i++) {
        shmem_team_destroy(tm[i]);
    }
}

enum { NR = 5, NB2B = 10 };

/* sbuf/dbuf hold NR elements of any reduction type (long double is largest) */
static void *sbuf, *dbuf;

/* kinds: 0 and, 1 or, 2 xor, 3 min/max/sum, 4 prod */
static long long gen(int kind, int i, int k, int rep)
{
    int b = (i * 5 + k + rep * 3) % 8;

    switch (kind) {
    case 0:
        return (long long) ~(1ULL << b);
    case 1:
        return (long long) (1ULL << b);
    case 2:
        return (long long) ((i + 1) * 0x9E3779B1ULL
                            + (k + 1) * (rep + 1) * 0x85EBCA6BULL);
    case 3:
        return (long long) ((i * 5 + k * 3 + rep) % 7) - 3;
    default:
        return (long long) ((i + k + rep) % 3) + 1;
    }
}

#define F_AND(a, b) ((a) & (b))
#define F_OR(a, b)  ((a) | (b))
#define F_XOR(a, b) ((a) ^ (b))
#define F_MIN(a, b) ((a) < (b) ? (a) : (b))
#define F_MAX(a, b) ((a) > (b) ? (a) : (b))
#define F_SUM(a, b) ((a) + (b))
#define F_PROD(a, b) ((a) * (b))

#define RB(TN, T, OPN, KIND, FOLD)                                            \
    do {                                                                      \
        T *s_ = (T *) sbuf, *d_ = (T *) dbuf;                                 \
        for (rep = 0; rep < 2; rep++) {                                       \
            shmem_sync_all();                                                 \
            if (member) {                                                     \
                for (k = 0; k < NR; k++) {                                    \
                    s_[k] = (T) gen(KIND, tme, k, rep);                       \
                    d_[k] = (T) -37;                                          \
                }                                                             \
            }                                                                 \
            shmem_sync_all();                                                 \
            if (member) {                                                     \
                rc = shmem_##TN##_##OPN##_reduce(team, d_, s_, NR);           \
                if (0 != rc) {                                                \
                    FAIL("%s %s_%s returned %d, expected 0", tname[c], #TN,   \
                         #OPN, rc);                                           \
                }                                                             \
                for (k = 0; k < NR; k++) {                                    \
                    T e = (T) gen(KIND, 0, k, rep);                           \
                    for (r = 1; r < tn; r++) {                                \
                        T v = (T) gen(KIND, r, k, rep);                       \
                        e = (T) FOLD(e, v);                                   \
                    }                                                         \
                    if (d_[k] != e) {                                         \
                        FAIL("%s %s_%s rep %d elem %d: expected %.17Lg, "     \
                             "got %.17Lg", tname[c], #TN, #OPN, rep, k,       \
                             (long double) e, (long double) d_[k]);           \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while (0)

#define BITWISE_TYPES(OPN, KIND, FOLD)                                        \
    RB(uchar, unsigned char, OPN, KIND, FOLD);                                \
    RB(ushort, unsigned short, OPN, KIND, FOLD);                              \
    RB(uint, unsigned int, OPN, KIND, FOLD);                                  \
    RB(ulong, unsigned long, OPN, KIND, FOLD);                                \
    RB(ulonglong, unsigned long long, OPN, KIND, FOLD);                       \
    RB(uint8, uint8_t, OPN, KIND, FOLD);                                      \
    RB(uint16, uint16_t, OPN, KIND, FOLD);                                    \
    RB(uint32, uint32_t, OPN, KIND, FOLD);                                    \
    RB(uint64, uint64_t, OPN, KIND, FOLD);                                    \
    RB(size, size_t, OPN, KIND, FOLD)

#define ARITH_TYPES(OPN, KIND, FOLD)                                          \
    RB(schar, signed char, OPN, KIND, FOLD);                                  \
    RB(short, short, OPN, KIND, FOLD);                                        \
    RB(int, int, OPN, KIND, FOLD);                                            \
    RB(long, long, OPN, KIND, FOLD);                                          \
    RB(longlong, long long, OPN, KIND, FOLD);                                 \
    RB(int8, int8_t, OPN, KIND, FOLD);                                        \
    RB(int16, int16_t, OPN, KIND, FOLD);                                      \
    RB(int32, int32_t, OPN, KIND, FOLD);                                      \
    RB(int64, int64_t, OPN, KIND, FOLD);                                      \
    RB(uint, unsigned int, OPN, KIND, FOLD);                                  \
    RB(ulonglong, unsigned long long, OPN, KIND, FOLD);                       \
    RB(size, size_t, OPN, KIND, FOLD);                                        \
    RB(ptrdiff, ptrdiff_t, OPN, KIND, FOLD);                                  \
    RB(float, float, OPN, KIND, FOLD);                                        \
    RB(double, double, OPN, KIND, FOLD);                                      \
    RB(longdouble, long double, OPN, KIND, FOLD)

/* products of up to 8 values in 1..3 overflow 8-bit types, so skip those */
#define PROD_TYPES(OPN, KIND, FOLD)                                           \
    RB(short, short, OPN, KIND, FOLD);                                        \
    RB(int, int, OPN, KIND, FOLD);                                            \
    RB(long, long, OPN, KIND, FOLD);                                          \
    RB(longlong, long long, OPN, KIND, FOLD);                                 \
    RB(int16, int16_t, OPN, KIND, FOLD);                                      \
    RB(int32, int32_t, OPN, KIND, FOLD);                                      \
    RB(int64, int64_t, OPN, KIND, FOLD);                                      \
    RB(uint, unsigned int, OPN, KIND, FOLD);                                  \
    RB(ulong, unsigned long, OPN, KIND, FOLD);                                \
    RB(float, float, OPN, KIND, FOLD);                                        \
    RB(double, double, OPN, KIND, FOLD);                                      \
    RB(longdouble, long double, OPN, KIND, FOLD)

static void all_ops(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    int rep, k, r, rc;

    BITWISE_TYPES(and, 0, F_AND);
    BITWISE_TYPES(or, 1, F_OR);
    BITWISE_TYPES(xor, 2, F_XOR);
    ARITH_TYPES(min, 3, F_MIN);
    ARITH_TYPES(max, 3, F_MAX);
    ARITH_TYPES(sum, 3, F_SUM);
    PROD_TYPES(prod, 4, F_PROD);
}

/*
 * NB2B reductions in a row on one team with no intervening synchronization,
 * each with its own source and destination, alternating operations.
 */
static void back_to_back(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    long *s = (long *) shmem_malloc(NB2B * NR * sizeof(long));
    long *d = (long *) shmem_malloc(NB2B * NR * sizeof(long));
    int j, k, r, rc;

    if (NULL == s || NULL == d) {
        FAIL("shmem_malloc failed");
        return;
    }
    for (j = 0; j < NB2B; j++) {
        for (k = 0; k < NR; k++) {
            s[j * NR + k] = member ? (long) gen(3, tme, k, j) : 0;
            d[j * NR + k] = -37;
        }
    }
    shmem_sync_all();
    if (member) {
        for (j = 0; j < NB2B; j++) {
            if (0 == j % 3) {
                rc = shmem_long_sum_reduce(team, d + j * NR, s + j * NR, NR);
            } else if (1 == j % 3) {
                rc = shmem_long_max_reduce(team, d + j * NR, s + j * NR, NR);
            } else {
                rc = shmem_long_min_reduce(team, d + j * NR, s + j * NR, NR);
            }
            if (0 != rc) {
                FAIL("%s back-to-back call %d returned %d", tname[c], j, rc);
            }
        }
        for (j = 0; j < NB2B; j++) {
            for (k = 0; k < NR; k++) {
                long e = (long) gen(3, 0, k, j);
                for (r = 1; r < tn; r++) {
                    long v = (long) gen(3, r, k, j);
                    if (0 == j % 3) {
                        e += v;
                    } else if (1 == j % 3) {
                        e = (v > e) ? v : e;
                    } else {
                        e = (v < e) ? v : e;
                    }
                }
                if (d[j * NR + k] != e) {
                    FAIL("%s back-to-back call %d elem %d: expected %ld, "
                         "got %ld", tname[c], j, k, e, d[j * NR + k]);
                }
            }
        }
    }
    shmem_sync_all();
    shmem_free(d);
    shmem_free(s);
}

int main(void)
{
    int c, n, pe, failed;

    shmem_init();
    pe = shmem_my_pe();
    n = shmem_n_pes();
    if (n < 2) {
        if (0 == pe) {
            printf("SKIP: need at least 2 PEs\n");
        }
        shmem_finalize();
        return 77;
    }

    sbuf = shmem_malloc(NR * sizeof(long double));
    dbuf = shmem_malloc(NR * sizeof(long double));
    if (NULL == sbuf || NULL == dbuf) {
        FAIL("shmem_malloc failed");
        shmem_global_exit(1);
    }
    make_teams(n);

    for (c = 0; c < NCASE; c++) {
        all_ops(c);
        back_to_back(c);
    }

    shmem_sync_all();
    destroy_teams();
    shmem_free(dbuf);
    shmem_free(sbuf);
    failed = (0 != nfail);
    shmem_finalize();
    return failed ? 1 : 0;
}
