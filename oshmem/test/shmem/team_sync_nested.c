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
 * shmem_team_sync on nested sub-teams: teams split from strided teams.
 * Start and stride of the grandchild are relative to the parent team.
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

/* team ordering test: put to next team member, sync, verify */
static void ring_check(shmem_team_t team, int *buf, int tag, int me)
{
    int tme = shmem_team_my_pe(team);
    int tn = shmem_team_n_pes(team);
    int next = shmem_team_translate_pe(team, (tme + 1) % tn, SHMEM_TEAM_WORLD);
    int prev = (tme + tn - 1) % tn;
    int rc;

    *buf = -1;
    rc = shmem_team_sync(team);
    CHECK(0 == rc, me, "nested sync (reset) return", 0, rc);
    shmem_int_p(buf, tag + tme, next);
    rc = shmem_team_sync(team);
    CHECK(0 == rc, me, "nested sync return", 0, rc);
    CHECK(*buf == tag + prev, me, "nested ring value", tag + prev, *buf);
}

int main(void)
{
    int me, npes, rc, i;
    int *buf;
    shmem_team_t t1 = SHMEM_TEAM_INVALID, t2 = SHMEM_TEAM_INVALID;
    shmem_team_t t3 = SHMEM_TEAM_INVALID, t4 = SHMEM_TEAM_INVALID;

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

    /* t1: world PEs 1..7 (7 members); t2: t1 members 0,2,4,6 = world 1,3,5,7;
     * t3: t2 members 1,2 (stride 1) = world 3,5; t4: t1 members 1,3,5 =
     * world 2,4,6 */
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 1, 1, 7, NULL, 0, &t1);
    if (me >= 1 && me <= 7) {
        CHECK(0 == rc, me, "t1 return", 0, rc);
    }
    CHECK((me >= 1 && me <= 7) == (t1 != SHMEM_TEAM_INVALID), me,
          "t1 validity", me >= 1 && me <= 7, t1 != SHMEM_TEAM_INVALID);

    rc = shmem_team_split_strided(t1, 0, 2, 4, NULL, 0, &t2);
    if (t1 == SHMEM_TEAM_INVALID) {
        CHECK(t2 == SHMEM_TEAM_INVALID, me, "t2 from invalid parent", 1, 0);
    } else {
        int member = (me % 2 == 1);
        CHECK(member == (t2 != SHMEM_TEAM_INVALID), me, "t2 validity",
              member, t2 != SHMEM_TEAM_INVALID);
        CHECK(shmem_team_n_pes(t2) == (member ? 4 : -1), me, "t2 n_pes",
              member ? 4 : -1, shmem_team_n_pes(t2));
        CHECK(shmem_team_my_pe(t2) == (member ? (me - 1) / 2 : -1), me,
              "t2 my_pe", member ? (me - 1) / 2 : -1, shmem_team_my_pe(t2));
    }

    rc = shmem_team_split_strided(t2, 1, 1, 2, NULL, 0, &t3);
    if (t2 == SHMEM_TEAM_INVALID) {
        CHECK(t3 == SHMEM_TEAM_INVALID, me, "t3 from invalid parent", 1, 0);
    } else {
        int member = (me == 3 || me == 5);
        CHECK(member == (t3 != SHMEM_TEAM_INVALID), me, "t3 validity",
              member, t3 != SHMEM_TEAM_INVALID);
        if (member) {
            CHECK(shmem_team_my_pe(t3) == (me == 3 ? 0 : 1), me, "t3 my_pe",
                  me == 3 ? 0 : 1, shmem_team_my_pe(t3));
            CHECK(shmem_team_translate_pe(t3, 1, SHMEM_TEAM_WORLD) == 5, me,
                  "t3 translate", 5, shmem_team_translate_pe(t3, 1,
                                                             SHMEM_TEAM_WORLD));
            CHECK(shmem_team_translate_pe(t3, 0, t2) == 1, me,
                  "t3->t2 translate", 1, shmem_team_translate_pe(t3, 0, t2));
        }
    }

    rc = shmem_team_split_strided(t1, 1, 2, 3, NULL, 0, &t4);
    if (t1 != SHMEM_TEAM_INVALID) {
        int member = (me == 2 || me == 4 || me == 6);
        CHECK(member == (t4 != SHMEM_TEAM_INVALID), me, "t4 validity",
              member, t4 != SHMEM_TEAM_INVALID);
        if (member) {
            CHECK(0 == rc, me, "t4 return", 0, rc);
            CHECK(shmem_team_my_pe(t4) == (me - 2) / 2, me, "t4 my_pe",
                  (me - 2) / 2, shmem_team_my_pe(t4));
        }
    }

    /* sync every level, and interleave syncs of different teams */
    for (i = 0; i < 5; i++) {
        if (t1 != SHMEM_TEAM_INVALID) ring_check(t1, buf, 1000 * (i + 1), me);
        if (t2 != SHMEM_TEAM_INVALID) ring_check(t2, buf, 2000 * (i + 1), me);
        if (t3 != SHMEM_TEAM_INVALID) ring_check(t3, buf, 3000 * (i + 1), me);
        if (t4 != SHMEM_TEAM_INVALID) ring_check(t4, buf, 4000 * (i + 1), me);
        rc = shmem_team_sync(SHMEM_TEAM_WORLD);
        CHECK(0 == rc, me, "world sync return", 0, rc);
    }

    /* destroying a parent does not affect the other teams' syncs of
     * independent children, so destroy children first */
    shmem_team_destroy(t4);
    shmem_team_destroy(t3);
    shmem_team_destroy(t2);
    shmem_team_destroy(t1);

    shmem_barrier_all();
    shmem_free(buf);
    shmem_finalize();
    return errors ? 1 : 0;
}
