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
 * Collectives on several teams interleaved: reduce on one team, broadcast
 * on another and fcollect/collect on a third, repeatedly and without
 * intervening synchronization.  Each team must keep its own pSync state.
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

enum { NITER = 8, NE = 3 };

static int gen(int i, int k, int it)
{
    return (i * 13 + k * 5 + it * 3 + 1) % 50;
}

static int tme_of(int c)
{
    return (SHMEM_TEAM_INVALID != tm[c]) ? shmem_team_my_pe(tm[c]) : -1;
}

static int tn_of(int c)
{
    return (SHMEM_TEAM_INVALID != tm[c]) ? shmem_team_n_pes(tm[c]) : 0;
}

/*
 * Iteration it issues, in this order on every member: sum reduce on team a,
 * broadcast on team b (root it % size), fcollect on team a, collect on
 * team b, max reduce on team b, all on separate buffers.
 */
static void interleave(int a, int b)
{
    enum { COL = NITER * 8 * NE };
    long *rs = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *rd = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *ms = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *md = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *bs = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *bd = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *fs = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *fd = (long *) shmem_malloc(COL * sizeof(long));
    long *cs = (long *) shmem_malloc(NITER * NE * sizeof(long));
    long *cd = (long *) shmem_malloc(COL * sizeof(long));
    int ina = (SHMEM_TEAM_INVALID != tm[a]);
    int inb = (SHMEM_TEAM_INVALID != tm[b]);
    int ra = tme_of(a), rb = tme_of(b);
    int na = tn_of(a), nb = tn_of(b);
    int it, k, r, rc;

    if (!rs || !rd || !ms || !md || !bs || !bd || !fs || !fd || !cs || !cd) {
        FAIL("shmem_malloc failed");
        return;
    }
    for (it = 0; it < NITER; it++) {
        for (k = 0; k < NE; k++) {
            rs[it * NE + k] = gen(ra + 1, k, it);
            ms[it * NE + k] = gen(rb + 1, k, it + 1);
            bs[it * NE + k] = (inb && rb == it % (nb ? nb : 1))
                              ? gen(99, k, it) : -5;
            fs[it * NE + k] = gen(ra + 1, k, it + 2);
            cs[it * NE + k] = gen(rb + 1, k, it + 3);
            rd[it * NE + k] = md[it * NE + k] = bd[it * NE + k] = -37;
        }
    }
    for (k = 0; k < COL; k++) {
        fd[k] = cd[k] = -37;
    }
    shmem_sync_all();

    for (it = 0; it < NITER; it++) {
        if (ina) {
            rc = shmem_long_sum_reduce(tm[a], rd + it * NE, rs + it * NE, NE);
            if (rc) {
                FAIL("%s/%s iter %d sum_reduce returned %d", tname[a],
                     tname[b], it, rc);
            }
        }
        if (inb) {
            rc = shmem_long_broadcast(tm[b], bd + it * NE, bs + it * NE, NE,
                                      it % nb);
            if (rc) {
                FAIL("%s/%s iter %d broadcast returned %d", tname[a],
                     tname[b], it, rc);
            }
        }
        if (ina) {
            rc = shmem_long_fcollect(tm[a], fd + it * 8 * NE, fs + it * NE,
                                     NE);
            if (rc) {
                FAIL("%s/%s iter %d fcollect returned %d", tname[a], tname[b],
                     it, rc);
            }
        }
        if (inb) {
            rc = shmem_long_collect(tm[b], cd + it * 8 * NE, cs + it * NE,
                                    (size_t) (rb % 3) + 1);
            if (rc) {
                FAIL("%s/%s iter %d collect returned %d", tname[a], tname[b],
                     it, rc);
            }
            rc = shmem_long_max_reduce(tm[b], md + it * NE, ms + it * NE, NE);
            if (rc) {
                FAIL("%s/%s iter %d max_reduce returned %d", tname[a],
                     tname[b], it, rc);
            }
        }
    }

    for (it = 0; it < NITER; it++) {
        for (k = 0; k < NE; k++) {
            long e;
            size_t off = 0, q;

            if (ina) {
                e = 0;
                for (r = 0; r < na; r++) {
                    e += gen(r + 1, k, it);
                }
                if (rd[it * NE + k] != e) {
                    FAIL("%s/%s iter %d sum elem %d: expected %ld, got %ld",
                         tname[a], tname[b], it, k, e, rd[it * NE + k]);
                }
                for (r = 0; r < na; r++) {
                    e = gen(r + 1, k, it + 2);
                    if (fd[it * 8 * NE + r * NE + k] != e) {
                        FAIL("%s/%s iter %d fcollect block %d elem %d: "
                             "expected %ld, got %ld", tname[a], tname[b], it,
                             r, k, e, fd[it * 8 * NE + r * NE + k]);
                    }
                }
            }
            if (inb) {
                e = gen(99, k, it);
                if (bd[it * NE + k] != e) {
                    FAIL("%s/%s iter %d broadcast elem %d: expected %ld, got "
                         "%ld", tname[a], tname[b], it, k, e,
                         bd[it * NE + k]);
                }
                e = gen(1, k, it + 1);
                for (r = 1; r < nb; r++) {
                    long v = gen(r + 1, k, it + 1);
                    e = (v > e) ? v : e;
                }
                if (md[it * NE + k] != e) {
                    FAIL("%s/%s iter %d max elem %d: expected %ld, got %ld",
                         tname[a], tname[b], it, k, e, md[it * NE + k]);
                }
                for (r = 0; r < nb; r++) {
                    for (q = 0; q < (size_t) (r % 3) + 1; q++) {
                        e = gen(r + 1, (int) q, it + 3);
                        if (cd[it * 8 * NE + off + q] != e) {
                            FAIL("%s/%s iter %d collect block %d elem %d: "
                                 "expected %ld, got %ld", tname[a], tname[b],
                                 it, r, (int) q, e,
                                 cd[it * 8 * NE + off + q]);
                        }
                    }
                    off += (size_t) (r % 3) + 1;
                }
            }
        }
    }
    shmem_sync_all();
    shmem_free(cd);
    shmem_free(cs);
    shmem_free(fd);
    shmem_free(fs);
    shmem_free(bd);
    shmem_free(bs);
    shmem_free(md);
    shmem_free(ms);
    shmem_free(rd);
    shmem_free(rs);
}

int main(void)
{
    int a, b, n, pe, failed;

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
    make_teams(n);

    /* every ordered pair of distinct team cases */
    for (a = 0; a < NCASE; a++) {
        for (b = 0; b < NCASE; b++) {
            if (a != b) {
                interleave(a, b);
            }
        }
    }

    shmem_sync_all();
    destroy_teams();
    failed = (0 != nfail);
    shmem_finalize();
    return failed ? 1 : 0;
}
