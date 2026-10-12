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
 * shmem_team_sync on SHMEM_TEAM_WORLD: int return value 0, team queries on
 * the world team, and ordering of puts issued before the sync.
 *
 * NP: 4
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

int main(void)
{
    int me, npes, i, rc;
    int *val;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 2) {
        shmem_finalize();
        return 77;
    }

    CHECK(SHMEM_TEAM_WORLD != SHMEM_TEAM_INVALID, me,
          "WORLD != INVALID", 1, 0);
    CHECK(shmem_team_my_pe(SHMEM_TEAM_WORLD) == me, me, "team_my_pe(WORLD)",
          me, shmem_team_my_pe(SHMEM_TEAM_WORLD));
    CHECK(shmem_team_n_pes(SHMEM_TEAM_WORLD) == npes, me,
          "team_n_pes(WORLD)", npes, shmem_team_n_pes(SHMEM_TEAM_WORLD));
    CHECK(shmem_team_my_pe(SHMEM_TEAM_INVALID) == -1, me,
          "team_my_pe(INVALID)", -1, shmem_team_my_pe(SHMEM_TEAM_INVALID));
    CHECK(shmem_team_n_pes(SHMEM_TEAM_INVALID) == -1, me,
          "team_n_pes(INVALID)", -1, shmem_team_n_pes(SHMEM_TEAM_INVALID));
    for (i = 0; i < npes; i++) {
        int t = shmem_team_translate_pe(SHMEM_TEAM_WORLD, i, SHMEM_TEAM_WORLD);
        CHECK(t == i, me, "translate_pe(WORLD, i, WORLD)", i, t);
    }

    val = shmem_malloc(sizeof(int));
    if (NULL == val) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }

    /* the return value must be an int, and 0 on success */
    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    CHECK(0 == rc, me, "shmem_team_sync return", 0, rc);

    /* puts before the sync are visible to everybody after it */
    for (i = 1; i <= 20; i++) {
        *val = -1;
        rc = shmem_team_sync(SHMEM_TEAM_WORLD);
        CHECK(0 == rc, me, "sync (reset) return", 0, rc);
        shmem_int_p(val, i * 1000 + me, (me + 1) % npes);
        rc = shmem_team_sync(SHMEM_TEAM_WORLD);
        CHECK(0 == rc, me, "sync return", 0, rc);
        CHECK(*val == i * 1000 + (me + npes - 1) % npes, me,
              "value from left neighbor", i * 1000 + (me + npes - 1) % npes,
              *val);
    }

    shmem_barrier_all();
    shmem_free(val);
    shmem_finalize();
    return errors ? 1 : 0;
}
