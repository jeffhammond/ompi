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
 * shmem_team_split_2d with xrange smaller than, equal to and larger than
 * the number of PEs, with and without npes divisible by xrange.  Row and
 * column teams are checked (size, rank, translate) and synchronized.
 *
 * NP: 7
 */

#include <stdio.h>
#include <shmem.h>

static int errors = 0;

#define CHECK(cond, pe, what, exp, act)                                     \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL: PE %d: %s: expected %d, got %d\n",                \
                   (pe), (what), (int) (exp), (int) (act));                 \
            errors++;                                                       \
        }                                                                   \
    } while (0)

static void check_2d(shmem_team_t parent, int pn, int pme, int xrange,
                     int me, int *buf)
{
    shmem_team_t xt = SHMEM_TEAM_INVALID, yt = SHMEM_TEAM_INVALID;
    int rc, i, xr = xrange > pn ? pn : xrange;
    int px = pme % xr, py = pme / xr;
    int xsize = (pn - py * xr < xr) ? pn - py * xr : xr;
    int ysize = (pn - px + xr - 1) / xr;

    rc = shmem_team_split_2d(parent, xrange, NULL, 0, &xt, NULL, 0, &yt);
    CHECK(0 == rc, me, "split_2d return", 0, rc);
    CHECK(xt != SHMEM_TEAM_INVALID && yt != SHMEM_TEAM_INVALID, me,
          "2d teams valid", 1, 0);
    if (xt == SHMEM_TEAM_INVALID || yt == SHMEM_TEAM_INVALID) {
        return;
    }

    CHECK(shmem_team_n_pes(xt) == xsize, me, "x n_pes", xsize,
          shmem_team_n_pes(xt));
    CHECK(shmem_team_my_pe(xt) == px, me, "x my_pe", px,
          shmem_team_my_pe(xt));
    CHECK(shmem_team_n_pes(yt) == ysize, me, "y n_pes", ysize,
          shmem_team_n_pes(yt));
    CHECK(shmem_team_my_pe(yt) == py, me, "y my_pe", py,
          shmem_team_my_pe(yt));

    /* global PE of x member i is me - px + i; of y member j is px + j*xr
     * when the parent is the world team */
    if (parent == SHMEM_TEAM_WORLD) {
        for (i = 0; i < xsize; i++) {
            int w = shmem_team_translate_pe(xt, i, SHMEM_TEAM_WORLD);
            CHECK(w == py * xr + i, me, "x translate", py * xr + i, w);
        }
        for (i = 0; i < ysize; i++) {
            int w = shmem_team_translate_pe(yt, i, SHMEM_TEAM_WORLD);
            CHECK(w == px + i * xr, me, "y translate", px + i * xr, w);
        }
    }

    /* each team member puts its team rank into slot of the next member,
     * the value must be visible after the sync */
    {
        int tme = shmem_team_my_pe(xt), tn = shmem_team_n_pes(xt);
        int nxt = shmem_team_translate_pe(xt, (tme + 1) % tn,
                                          SHMEM_TEAM_WORLD);
        *buf = -1;
        rc = shmem_team_sync(xt);
        CHECK(0 == rc, me, "x sync (reset) return", 0, rc);
        shmem_int_p(buf, 10 + tme, nxt);
        rc = shmem_team_sync(xt);
        CHECK(0 == rc, me, "x sync return", 0, rc);
        CHECK(*buf == 10 + (tme + tn - 1) % tn, me, "x ring value",
              10 + (tme + tn - 1) % tn, *buf);
    }
    {
        int tme = shmem_team_my_pe(yt), tn = shmem_team_n_pes(yt);
        int nxt = shmem_team_translate_pe(yt, (tme + 1) % tn,
                                          SHMEM_TEAM_WORLD);
        *buf = -1;
        rc = shmem_team_sync(yt);
        CHECK(0 == rc, me, "y sync (reset) return", 0, rc);
        shmem_int_p(buf, 50 + tme, nxt);
        rc = shmem_team_sync(yt);
        CHECK(0 == rc, me, "y sync return", 0, rc);
        CHECK(*buf == 50 + (tme + tn - 1) % tn, me, "y ring value",
              50 + (tme + tn - 1) % tn, *buf);
    }

    shmem_team_destroy(yt);
    shmem_team_destroy(xt);
}

int main(void)
{
    int me, npes, xr;
    int *buf;
    shmem_team_t sub = SHMEM_TEAM_INVALID;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 7) {
        shmem_finalize();
        return 77;
    }
    buf = shmem_malloc(sizeof(int));
    if (NULL == buf) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }

    /* world: 1 (all in one column), divisors, non-divisors, == npes,
     * and > npes */
    for (xr = 1; xr <= npes + 3; xr++) {
        check_2d(SHMEM_TEAM_WORLD, npes, me, xr, me, buf);
        shmem_barrier_all();
    }

    /* 2D split of a strided parent (world PEs 1..5, 5 members) */
    shmem_team_split_strided(SHMEM_TEAM_WORLD, 1, 1, 5, NULL, 0, &sub);
    if (sub != SHMEM_TEAM_INVALID) {
        for (xr = 2; xr <= 7; xr++) {
            check_2d(sub, 5, shmem_team_my_pe(sub), xr, me, buf);
        }
    } else {
        CHECK(me == 0 || me > 5, me, "member of sub parent", 0, 1);
    }
    shmem_team_destroy(sub);

    shmem_barrier_all();
    shmem_free(buf);
    shmem_finalize();
    return errors ? 1 : 0;
}
