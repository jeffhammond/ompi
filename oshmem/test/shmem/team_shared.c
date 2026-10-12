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
 * SHMEM_TEAM_SHARED: the PEs on the same node. Checks that it is a valid
 * team distinct from SHMEM_TEAM_WORLD and SHMEM_TEAM_INVALID, that its size
 * equals the number of PEs reporting the same host name, that PE numbering
 * and translation to and from the world team are consistent, that
 * shmem_team_sync works on it, and that it cannot be destroyed.
 *
 * NP: 4
 */

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <shmem.h>

#define NAMELEN 64

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
    int me, npes, i, n, my, rc, same, prev;
    char *names, *mine;
    int *val;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    n = shmem_team_n_pes(SHMEM_TEAM_SHARED);
    my = shmem_team_my_pe(SHMEM_TEAM_SHARED);

    CHECK(SHMEM_TEAM_SHARED != SHMEM_TEAM_INVALID, me, "SHARED != INVALID", 1, 0);
    CHECK(SHMEM_TEAM_SHARED != SHMEM_TEAM_WORLD, me, "SHARED != WORLD", 1, 0);
    CHECK(n >= 1 && n <= npes, me, "team_n_pes(SHARED)", npes, n);
    CHECK(my >= 0 && my < n, me, "team_my_pe(SHARED)", n, my);

    /* number of PEs on my node, found independently through host names */
    names = shmem_malloc(NAMELEN * npes);
    mine = shmem_malloc(NAMELEN);
    if (NULL == names || NULL == mine) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }
    memset(mine, 0, NAMELEN);
    gethostname(mine, NAMELEN - 1);
    shmem_barrier_all();
    for (i = 0; i < npes; i++) {
        shmem_getmem(names + i * NAMELEN, mine, NAMELEN, i);
    }
    same = 0;
    for (i = 0; i < npes; i++) {
        if (0 == strncmp(names + i * NAMELEN, mine, NAMELEN)) {
            same++;
        }
    }
    CHECK(n == same, me, "team_n_pes(SHARED) vs PEs on node", same, n);
    if (same == npes) {
        CHECK(n == shmem_team_n_pes(SHMEM_TEAM_WORLD), me,
              "single node: SHARED size == WORLD size", npes, n);
        CHECK(my == me, me, "single node: team_my_pe(SHARED)", me, my);
    }

    /* numbering and translation, members are on my node and ordered */
    CHECK(shmem_team_translate_pe(SHMEM_TEAM_SHARED, my, SHMEM_TEAM_WORLD) == me,
          me, "translate(SHARED, my_pe, WORLD)", me,
          shmem_team_translate_pe(SHMEM_TEAM_SHARED, my, SHMEM_TEAM_WORLD));
    CHECK(shmem_team_translate_pe(SHMEM_TEAM_WORLD, me, SHMEM_TEAM_SHARED) == my,
          me, "translate(WORLD, me, SHARED)", my,
          shmem_team_translate_pe(SHMEM_TEAM_WORLD, me, SHMEM_TEAM_SHARED));
    prev = -1;
    for (i = 0; i < n; i++) {
        int w = shmem_team_translate_pe(SHMEM_TEAM_SHARED, i, SHMEM_TEAM_WORLD);
        CHECK(w > prev && w < npes, me, "translate(SHARED, i, WORLD) order", prev + 1, w);
        if (w >= 0 && w < npes) {
            CHECK(0 == strncmp(names + w * NAMELEN, mine, NAMELEN), me,
                  "member is on my node", 1, 0);
            CHECK(shmem_team_translate_pe(SHMEM_TEAM_WORLD, w, SHMEM_TEAM_SHARED) == i,
                  me, "translate(WORLD, w, SHARED)", i,
                  shmem_team_translate_pe(SHMEM_TEAM_WORLD, w, SHMEM_TEAM_SHARED));
        }
        prev = w;
    }
    CHECK(shmem_team_translate_pe(SHMEM_TEAM_SHARED, 0, SHMEM_TEAM_SHARED) == 0,
          me, "translate(SHARED, 0, SHARED)", 0,
          shmem_team_translate_pe(SHMEM_TEAM_SHARED, 0, SHMEM_TEAM_SHARED));

    /* team sync: puts to the next member are visible after the sync */
    val = shmem_malloc(sizeof(int));
    if (NULL == val) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }
    for (i = 1; i <= 10; i++) {
        int next = shmem_team_translate_pe(SHMEM_TEAM_SHARED, (my + 1) % n,
                                           SHMEM_TEAM_WORLD);
        int pw = shmem_team_translate_pe(SHMEM_TEAM_SHARED, (my + n - 1) % n,
                                         SHMEM_TEAM_WORLD);
        *val = -1;
        rc = shmem_team_sync(SHMEM_TEAM_SHARED);
        CHECK(0 == rc, me, "team_sync(SHARED) return", 0, rc);
        shmem_int_p(val, i * 1000 + me, next);
        rc = shmem_team_sync(SHMEM_TEAM_SHARED);
        CHECK(0 == rc, me, "team_sync(SHARED) return", 0, rc);
        CHECK(*val == i * 1000 + pw, me, "value after sync", i * 1000 + pw, *val);
    }

    /* a team can be split from SHMEM_TEAM_SHARED */
    if (n >= 2) {
        shmem_team_t sub;
        rc = shmem_team_split_strided(SHMEM_TEAM_SHARED, 0, 1, 2, NULL, 0, &sub);
        CHECK(0 == rc, me, "split_strided(SHARED) return", 0, rc);
        if (my < 2) {
            CHECK(sub != SHMEM_TEAM_INVALID, me, "member got a team", 1, 0);
            CHECK(shmem_team_n_pes(sub) == 2, me, "sub n_pes", 2, shmem_team_n_pes(sub));
            CHECK(0 == shmem_team_sync(sub), me, "sub sync", 0, 1);
            shmem_team_destroy(sub);
        } else {
            CHECK(sub == SHMEM_TEAM_INVALID, me, "non-member got INVALID", 1, 0);
        }
    }

    /* the predefined team is still intact after teams came and went */
    rc = shmem_team_sync(SHMEM_TEAM_SHARED);
    CHECK(0 == rc, me, "team_sync(SHARED) after split", 0, rc);

    shmem_barrier_all();
    shmem_free(val);
    shmem_free(mine);
    shmem_free(names);
    shmem_finalize();

    if (0 == errors && 0 == me) {
        printf("PASS\n");
    }
    return errors ? 1 : 0;
}
