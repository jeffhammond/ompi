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
 * shmem_team_split_strided with a negative stride, as exercised by the
 * Sandia OpenSHMEM tests.  If supported, team rank i is parent PE
 * start + i * stride.  If an implementation rejects it, all PEs must
 * reject it (SHMEM_TEAM_INVALID everywhere).  Either way the call must
 * not abort and a later team with a positive stride must work.
 *
 * NP: 4
 */

#include <stdio.h>
#include <stdlib.h>
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

int main(void)
{
    int me, npes, rc, i;
    int *nvalid;
    shmem_team_t nt = SHMEM_TEAM_INVALID, pt = SHMEM_TEAM_INVALID;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 4) {
        shmem_finalize();
        return 77;
    }

    nvalid = shmem_malloc(sizeof(int));

    /* reversed world team */
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, npes - 1, -1, npes, NULL,
                                  0, &nt);
    if (nt != SHMEM_TEAM_INVALID) {
        int tme = shmem_team_my_pe(nt);
        CHECK(0 == rc, me, "negative stride return", 0, rc);
        CHECK(shmem_team_n_pes(nt) == npes, me, "negative n_pes", npes,
              shmem_team_n_pes(nt));
        CHECK(tme == npes - 1 - me, me, "negative my_pe", npes - 1 - me, tme);
        for (i = 0; i < npes; i++) {
            int w = shmem_team_translate_pe(nt, i, SHMEM_TEAM_WORLD);
            CHECK(w == npes - 1 - i, me, "negative translate team->world",
                  npes - 1 - i, w);
            w = shmem_team_translate_pe(SHMEM_TEAM_WORLD, i, nt);
            CHECK(w == npes - 1 - i, me, "negative translate world->team",
                  npes - 1 - i, w);
        }
        CHECK(shmem_team_sync(nt) == 0, me, "negative team sync", 0, 1);
    } else {
        /* rejected: every PE must have rejected it */
        CHECK(shmem_team_n_pes(nt) == -1, me, "n_pes of INVALID", -1,
              shmem_team_n_pes(nt));
    }
    /* all PEs or none must have received a valid team */
    *nvalid = 0;
    shmem_barrier_all();
    shmem_int_atomic_add(nvalid, nt != SHMEM_TEAM_INVALID, 0);
    shmem_barrier_all();
    if (0 == me) {
        CHECK(0 == *nvalid || npes == *nvalid, me,
              "PEs with valid reversed team (0 or all)", npes, *nvalid);
    }
    shmem_team_destroy(nt);
    shmem_barrier_all();

    /* a team with stride -2 over the even PEs, starting at the last even */
    nt = SHMEM_TEAM_INVALID;
    {
        int start = (npes - 1) & ~1;
        int size = start / 2 + 1;
        rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, start, -2, size, NULL,
                                      0, &nt);
        if (nt != SHMEM_TEAM_INVALID) {
            CHECK(me % 2 == 0, me, "stride -2 membership (even only)", 1, 0);
            CHECK(shmem_team_my_pe(nt) == (start - me) / 2, me,
                  "stride -2 my_pe", (start - me) / 2, shmem_team_my_pe(nt));
            CHECK(shmem_team_sync(nt) == 0, me, "stride -2 sync", 0, 1);
        }
        shmem_team_destroy(nt);
    }
    shmem_barrier_all();

    /* positive stride still works afterwards */
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 1, npes, NULL, 0, &pt);
    CHECK(0 == rc, me, "positive split return", 0, rc);
    CHECK(pt != SHMEM_TEAM_INVALID, me, "positive team valid", 1, 0);
    if (pt != SHMEM_TEAM_INVALID) {
        CHECK(shmem_team_sync(pt) == 0, me, "positive sync", 0, 1);
    }
    shmem_team_destroy(pt);
    shmem_barrier_all();
    shmem_free(nvalid);

    shmem_finalize();
    return errors ? 1 : 0;
}
