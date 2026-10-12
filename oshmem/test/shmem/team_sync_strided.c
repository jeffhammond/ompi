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
 * shmem_team_sync on strided sub-teams of the world team with various start
 * offsets and strides; non-members get SHMEM_TEAM_INVALID; team_my_pe,
 * team_n_pes and translate_pe are checked for members and non-members.
 *
 * NP: 8
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

static const int cfg[][3] = {
    /* start, stride, size */
    {0, 1, 8}, {0, 2, 4}, {1, 2, 4}, {0, 3, 3}, {1, 3, 3}, {2, 3, 2},
    {0, 4, 2}, {3, 4, 2}, {5, 1, 3}, {2, 1, 1}, {7, 1, 1}, {0, 7, 2},
    {1, 6, 2}
};

int main(void)
{
    int me, npes, c, i, rc;
    int *buf;
    int ncfg = (int) (sizeof(cfg) / sizeof(cfg[0]));

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 8) {
        shmem_finalize();
        return 77;
    }

    buf = shmem_malloc(sizeof(int));
    if (NULL == buf) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }

    for (c = 0; c < ncfg; c++) {
        int start = cfg[c][0], stride = cfg[c][1], size = cfg[c][2];
        int member = (me >= start) && (me < start + size * stride) &&
                     ((me - start) % stride == 0);
        int idx = member ? (me - start) / stride : -1;
        shmem_team_t team = SHMEM_TEAM_INVALID;

        rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, size,
                                      NULL, 0, &team);
        if (member) {
            CHECK(0 == rc, me, "split_strided return (member)", 0, rc);
            CHECK(team != SHMEM_TEAM_INVALID, me, "team valid for member",
                  1, 0);
        } else {
            CHECK(team == SHMEM_TEAM_INVALID, me,
                  "team INVALID for non-member", 1, 0);
        }

        CHECK(shmem_team_my_pe(team) == idx, me, "team_my_pe", idx,
              shmem_team_my_pe(team));
        CHECK(shmem_team_n_pes(team) == (member ? size : -1), me,
              "team_n_pes", member ? size : -1, shmem_team_n_pes(team));

        if (member) {
            for (i = 0; i < size; i++) {
                int w = shmem_team_translate_pe(team, i, SHMEM_TEAM_WORLD);
                CHECK(w == start + i * stride, me, "translate team->world",
                      start + i * stride, w);
            }
            for (i = 0; i < npes; i++) {
                int inm = (i >= start) && (i < start + size * stride) &&
                          ((i - start) % stride == 0);
                int exp = inm ? (i - start) / stride : -1;
                int t = shmem_team_translate_pe(SHMEM_TEAM_WORLD, i, team);
                CHECK(t == exp, me, "translate world->team", exp, t);
            }

            /* ordering: put to the next member, then sync */
            *buf = -1;
            rc = shmem_team_sync(team);
            CHECK(0 == rc, me, "team_sync (reset) return", 0, rc);
            shmem_int_p(buf, 100 * c + idx,
                        start + ((idx + 1) % size) * stride);
            rc = shmem_team_sync(team);
            CHECK(0 == rc, me, "team_sync return", 0, rc);
            CHECK(*buf == 100 * c + (idx + size - 1) % size, me,
                  "value from previous member",
                  100 * c + (idx + size - 1) % size, *buf);
            for (i = 0; i < 5; i++) {
                rc = shmem_team_sync(team);
                CHECK(0 == rc, me, "repeated team_sync return", 0, rc);
            }
        }
        shmem_team_destroy(team);
        shmem_barrier_all();
    }

    shmem_free(buf);
    shmem_finalize();
    return errors ? 1 : 0;
}
