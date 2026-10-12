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
 * Team fcollect and collect (with a different element count on each PE,
 * including zero) on SHMEM_TEAM_WORLD, strided teams and 2D-split teams.
 * The destination just past the gathered data must stay untouched.
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

enum { NF = 3, NMAX = 4, NK = 6 };

/* symmetric: up to NMAX elements per PE of the largest type, 8 PEs */
static void *sbuf, *dbuf;

static int gen(int i, int k, int rep)
{
    return (i * 11 + k * 3 + rep * 7 + 1) % 100;
}

/* element count contributed by team rank i in collect; 0 for i == 2 */
static size_t cnt(int i)
{
    return (size_t) ((i * 2 + 1) % 5);
}

#define COLLECT_TYPE(TN, T)                                                   \
    do {                                                                      \
        T *s_ = (T *) sbuf, *d_ = (T *) dbuf;                                 \
        size_t mine = member ? cnt(tme) : 0, total = 0, off = 0, q;           \
        for (r = 0; member && r < tn; r++) {                                  \
            total += cnt(r);                                                  \
        }                                                                     \
        shmem_sync_all();                                                     \
        for (k = 0; k < NMAX * 2 * 16; k++) {                                 \
            d_[k] = (T) 101;                                                  \
        }                                                                     \
        for (k = 0; k < NMAX; k++) {                                          \
            s_[k] = (T) gen(member ? tme : 0, k, 0);                          \
        }                                                                     \
        shmem_sync_all();                                                     \
        if (member) {                                                         \
            rc = shmem_##TN##_fcollect(team, d_, s_, NF);                     \
            if (0 != rc) {                                                    \
                FAIL("%s %s_fcollect returned %d", tname[c], #TN, rc);        \
            }                                                                 \
            for (r = 0; r < tn; r++) {                                        \
                for (k = 0; k < NF; k++) {                                    \
                    T e = (T) gen(r, k, 0);                                   \
                    if (d_[r * NF + k] != e) {                                \
                        FAIL("%s %s_fcollect (team rank %d) block %d elem "   \
                             "%d: expected %d, got %d", tname[c], #TN, tme,   \
                             r, k, (int) e, (int) d_[r * NF + k]);            \
                    }                                                         \
                }                                                             \
            }                                                                 \
            if (d_[tn * NF] != (T) 101) {                                     \
                FAIL("%s %s_fcollect (team rank %d) wrote past the "          \
                     "result", tname[c], #TN, tme);                           \
            }                                                                 \
        }                                                                     \
        shmem_sync_all();                                                     \
        for (k = 0; k < NMAX * 2 * 16; k++) {                                 \
            d_[k] = (T) 101;                                                  \
        }                                                                     \
        shmem_sync_all();                                                     \
        if (member) {                                                         \
            rc = shmem_##TN##_collect(team, d_, s_, mine);                    \
            if (0 != rc) {                                                    \
                FAIL("%s %s_collect returned %d", tname[c], #TN, rc);         \
            }                                                                 \
            for (r = 0; r < tn; r++) {                                        \
                for (q = 0; q < cnt(r); q++) {                                \
                    T e = (T) gen(r, (int) q, 0);                             \
                    if (d_[off + q] != e) {                                   \
                        FAIL("%s %s_collect (team rank %d) block %d elem "    \
                             "%d: expected %d, got %d", tname[c], #TN, tme,   \
                             r, (int) q, (int) e, (int) d_[off + q]);         \
                    }                                                         \
                }                                                             \
                off += cnt(r);                                                \
            }                                                                 \
            if (d_[total] != (T) 101) {                                       \
                FAIL("%s %s_collect (team rank %d) wrote past the "           \
                     "result", tname[c], #TN, tme);                           \
            }                                                                 \
        }                                                                     \
    } while (0)

static void all_types(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    int k, r, rc;

    COLLECT_TYPE(char, char);
    COLLECT_TYPE(short, short);
    COLLECT_TYPE(int, int);
    COLLECT_TYPE(long, long);
    COLLECT_TYPE(longlong, long long);
    COLLECT_TYPE(float, float);
    COLLECT_TYPE(double, double);
    COLLECT_TYPE(uint8, uint8_t);
    COLLECT_TYPE(int32, int32_t);
    COLLECT_TYPE(uint64, uint64_t);
    COLLECT_TYPE(size, size_t);
}

/* Untyped forms, in bytes. */
static void mem_forms(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    unsigned char *s = (unsigned char *) sbuf, *d = (unsigned char *) dbuf;
    size_t mine = member ? cnt(tme) * 3 : 0, off = 0, q;
    int k, r, rc;

    shmem_sync_all();
    for (k = 0; k < 256; k++) {
        d[k] = 0xEE;
    }
    for (k = 0; k < 16; k++) {
        s[k] = (unsigned char) gen(member ? tme : 0, k, 2);
    }
    shmem_sync_all();
    if (member) {
        rc = shmem_fcollectmem(team, d, s, 5);
        if (0 != rc) {
            FAIL("%s fcollectmem returned %d", tname[c], rc);
        }
        for (r = 0; r < tn; r++) {
            for (k = 0; k < 5; k++) {
                if (d[r * 5 + k] != (unsigned char) gen(r, k, 2)) {
                    FAIL("%s fcollectmem (team rank %d) block %d byte %d: "
                         "expected %d, got %d", tname[c], tme, r, k,
                         gen(r, k, 2), (int) d[r * 5 + k]);
                }
            }
        }
    }
    shmem_sync_all();
    for (k = 0; k < 256; k++) {
        d[k] = 0xEE;
    }
    shmem_sync_all();
    if (member) {
        rc = shmem_collectmem(team, d, s, mine);
        if (0 != rc) {
            FAIL("%s collectmem returned %d", tname[c], rc);
        }
        for (r = 0; r < tn; r++) {
            for (q = 0; q < cnt(r) * 3; q++) {
                if (d[off + q] != (unsigned char) gen(r, (int) q, 2)) {
                    FAIL("%s collectmem (team rank %d) block %d byte %d: "
                         "expected %d, got %d", tname[c], tme, r, (int) q,
                         gen(r, (int) q, 2), (int) d[off + q]);
                }
            }
            off += cnt(r) * 3;
        }
    }
}

/* NK fcollect and collect calls in a row without synchronization. */
static void back_to_back(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    enum { STRIDE = 8 * NMAX };
    int *s = (int *) shmem_malloc(NK * NMAX * sizeof(int));
    int *d = (int *) shmem_malloc(NK * STRIDE * sizeof(int));
    int j, k, r, rc;
    size_t off, q;

    if (NULL == s || NULL == d) {
        FAIL("shmem_malloc failed");
        return;
    }
    for (j = 0; j < NK; j++) {
        for (k = 0; k < NMAX; k++) {
            s[j * NMAX + k] = gen(member ? tme : 0, k, j);
        }
    }
    for (k = 0; k < NK * STRIDE; k++) {
        d[k] = 101;
    }
    shmem_sync_all();
    if (member) {
        for (j = 0; j < NK; j++) {
            if (0 == j % 2) {
                rc = shmem_int_fcollect(team, d + j * STRIDE, s + j * NMAX, 2);
            } else {
                rc = shmem_int_collect(team, d + j * STRIDE, s + j * NMAX,
                                       cnt(tme));
            }
            if (0 != rc) {
                FAIL("%s back-to-back call %d returned %d", tname[c], j, rc);
            }
        }
        for (j = 0; j < NK; j++) {
            off = 0;
            for (r = 0; r < tn; r++) {
                size_t m = (0 == j % 2) ? 2 : cnt(r);
                for (q = 0; q < m; q++) {
                    int e = gen(r, (int) q, j);
                    if (d[j * STRIDE + off + q] != e) {
                        FAIL("%s back-to-back call %d (team rank %d) block "
                             "%d elem %d: expected %d, got %d", tname[c], j,
                             tme, r, (int) q, e, d[j * STRIDE + off + q]);
                    }
                }
                off += m;
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

    /* d_ is indexed up to NMAX * 2 * 16 elements of the largest type (8 B) */
    sbuf = shmem_malloc(NMAX * 16);
    dbuf = shmem_malloc(NMAX * 2 * 16 * 8);
    if (NULL == sbuf || NULL == dbuf) {
        FAIL("shmem_malloc failed");
        shmem_global_exit(1);
    }
    make_teams(n);

    for (c = 0; c < NCASE; c++) {
        all_types(c);
        mem_forms(c);
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
