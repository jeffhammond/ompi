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
 * shmem_ctx_get_team and shmem_team_create_ctx: the default context maps to
 * SHMEM_TEAM_WORLD, a context created from a team maps back to that team,
 * and SHMEM_CTX_INVALID / SHMEM_TEAM_INVALID are reported as errors.
 *
 * NP: 4
 */

#include <stdio.h>
#include <shmem.h>

static int nfail = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            printf("FAIL: PE %d: line %d: ", shmem_my_pe(), __LINE__); \
            printf(__VA_ARGS__);                                  \
            printf("\n");                                         \
            nfail++;                                              \
        }                                                         \
    } while (0)

int main(void)
{
    shmem_team_t t, sub = SHMEM_TEAM_INVALID;
    shmem_ctx_t c, c2 = SHMEM_CTX_INVALID, cs = SHMEM_CTX_INVALID;
    int rc, me, npes;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 4) {
        shmem_finalize();
        return 77;
    }

    /* default context -> world */
    t = SHMEM_TEAM_INVALID;
    rc = shmem_ctx_get_team(SHMEM_CTX_DEFAULT, &t);
    CHECK(0 == rc, "get_team(default) rc=%d expected 0", rc);
    CHECK(SHMEM_TEAM_WORLD == t, "get_team(default) returned %p expected WORLD %p",
          (void *) t, (void *) SHMEM_TEAM_WORLD);

    /* invalid context */
    t = SHMEM_TEAM_WORLD;
    rc = shmem_ctx_get_team(SHMEM_CTX_INVALID, &t);
    CHECK(0 != rc, "get_team(SHMEM_CTX_INVALID) rc=0 expected nonzero");

    /* invalid team */
    c = SHMEM_CTX_DEFAULT;
    rc = shmem_team_create_ctx(SHMEM_TEAM_INVALID, 0, &c);
    CHECK(0 != rc, "create_ctx(SHMEM_TEAM_INVALID) rc=0 expected nonzero");
    CHECK(SHMEM_CTX_INVALID == c, "create_ctx(SHMEM_TEAM_INVALID) ctx=%p expected SHMEM_CTX_INVALID",
          (void *) c);

    /* context on the world team */
    c = SHMEM_CTX_INVALID;
    rc = shmem_team_create_ctx(SHMEM_TEAM_WORLD, 0, &c);
    CHECK(0 == rc, "create_ctx(WORLD) rc=%d expected 0", rc);
    CHECK(SHMEM_CTX_INVALID != c, "create_ctx(WORLD) returned SHMEM_CTX_INVALID");
    if (0 == rc && SHMEM_CTX_INVALID != c) {
        t = SHMEM_TEAM_INVALID;
        rc = shmem_ctx_get_team(c, &t);
        CHECK(0 == rc, "get_team(world ctx) rc=%d expected 0", rc);
        CHECK(SHMEM_TEAM_WORLD == t, "get_team(world ctx) returned %p expected WORLD %p",
              (void *) t, (void *) SHMEM_TEAM_WORLD);
    }

    /* context on a sub-team: every other PE */
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 2, (npes + 1) / 2, NULL, 0, &sub);
    CHECK(0 == rc, "split_strided rc=%d expected 0", rc);
    if (0 == me % 2) {
        CHECK(SHMEM_TEAM_INVALID != sub, "even PE got SHMEM_TEAM_INVALID from split");
        if (SHMEM_TEAM_INVALID != sub) {
            rc = shmem_team_create_ctx(sub, 0, &cs);
            CHECK(0 == rc, "create_ctx(sub) rc=%d expected 0", rc);
            CHECK(SHMEM_CTX_INVALID != cs, "create_ctx(sub) returned SHMEM_CTX_INVALID");
            if (0 == rc && SHMEM_CTX_INVALID != cs) {
                t = SHMEM_TEAM_INVALID;
                rc = shmem_ctx_get_team(cs, &t);
                CHECK(0 == rc, "get_team(sub ctx) rc=%d expected 0", rc);
                CHECK(sub == t, "get_team(sub ctx) returned %p expected sub %p",
                      (void *) t, (void *) sub);
                CHECK(SHMEM_TEAM_WORLD != t, "get_team(sub ctx) returned WORLD");
            }
            /* a second context from the same team, with options */
            rc = shmem_team_create_ctx(sub, SHMEM_CTX_PRIVATE, &c2);
            CHECK(0 == rc, "create_ctx(sub, PRIVATE) rc=%d expected 0", rc);
            if (0 == rc && SHMEM_CTX_INVALID != c2) {
                t = SHMEM_TEAM_INVALID;
                rc = shmem_ctx_get_team(c2, &t);
                CHECK(0 == rc && sub == t, "get_team(sub ctx 2) rc=%d team %p expected sub %p",
                      rc, (void *) t, (void *) sub);
            }
        }
    } else {
        CHECK(SHMEM_TEAM_INVALID == sub, "odd PE got a valid team from split");
    }

    /* destroy; a context re-created afterwards must not keep a stale team */
    if (SHMEM_CTX_INVALID != cs) {
        shmem_ctx_destroy(cs);
    }
    if (SHMEM_CTX_INVALID != c2) {
        shmem_ctx_destroy(c2);
    }
    cs = SHMEM_CTX_INVALID;
    rc = shmem_team_create_ctx(SHMEM_TEAM_WORLD, 0, &cs);
    CHECK(0 == rc, "re-create_ctx(WORLD) rc=%d expected 0", rc);
    if (0 == rc && SHMEM_CTX_INVALID != cs) {
        t = SHMEM_TEAM_INVALID;
        rc = shmem_ctx_get_team(cs, &t);
        CHECK(0 == rc && SHMEM_TEAM_WORLD == t,
              "get_team(re-created world ctx) rc=%d team %p expected WORLD %p",
              rc, (void *) t, (void *) SHMEM_TEAM_WORLD);
        shmem_ctx_destroy(cs);
    }
    if (SHMEM_CTX_INVALID != c) {
        shmem_ctx_destroy(c);
    }

    shmem_barrier_all();
    if (SHMEM_TEAM_INVALID != sub) {
        shmem_team_destroy(sub);
    }
    shmem_finalize();

    return nfail ? 1 : 0;
}
