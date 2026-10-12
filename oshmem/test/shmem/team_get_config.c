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
 * shmem_team_get_config returns the configuration given to
 * shmem_team_split_strided, copying only the fields selected by
 * config_mask, and reports SHMEM_TEAM_INVALID as an error.
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
    shmem_team_t a = SHMEM_TEAM_INVALID, b = SHMEM_TEAM_INVALID;
    shmem_team_config_t cfg, out;
    int rc, npes;

    shmem_init();
    npes = shmem_n_pes();
    if (npes < 4) {
        shmem_finalize();
        return 77;
    }

    /* split with num_contexts = 3 honoured via the mask */
    cfg.num_contexts = 3;
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 1, npes, &cfg,
                                  SHMEM_TEAM_NUM_CONTEXTS, &a);
    CHECK(0 == rc, "split with config rc=%d expected 0", rc);
    CHECK(SHMEM_TEAM_INVALID != a, "split returned SHMEM_TEAM_INVALID on a member");

    /* split with a different value and config_mask 0: the value is ignored */
    cfg.num_contexts = 5;
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 1, npes, &cfg, 0, &b);
    CHECK(0 == rc, "split without mask rc=%d expected 0", rc);

    if (SHMEM_TEAM_INVALID != a) {
        out.num_contexts = -77;
        rc = shmem_team_get_config(a, SHMEM_TEAM_NUM_CONTEXTS, &out);
        CHECK(0 == rc, "get_config(a, NUM_CONTEXTS) rc=%d expected 0", rc);
        CHECK(3 == out.num_contexts, "get_config(a) num_contexts=%d expected 3",
              out.num_contexts);

        /* mask 0: success, and nothing is copied */
        out.num_contexts = -77;
        rc = shmem_team_get_config(a, 0, &out);
        CHECK(0 == rc, "get_config(a, 0) rc=%d expected 0", rc);
        CHECK(-77 == out.num_contexts, "get_config(a, 0) modified config to %d",
              out.num_contexts);
    }

    if (SHMEM_TEAM_INVALID != b) {
        out.num_contexts = -77;
        rc = shmem_team_get_config(b, SHMEM_TEAM_NUM_CONTEXTS, &out);
        CHECK(0 == rc, "get_config(b, NUM_CONTEXTS) rc=%d expected 0", rc);
        CHECK(5 != out.num_contexts,
              "get_config(b) returned num_contexts=5 from a split that did not select it");
    }

    /* the world team is a valid argument */
    out.num_contexts = -77;
    rc = shmem_team_get_config(SHMEM_TEAM_WORLD, SHMEM_TEAM_NUM_CONTEXTS, &out);
    CHECK(0 == rc, "get_config(WORLD) rc=%d expected 0", rc);

    /* an invalid team is an error */
    rc = shmem_team_get_config(SHMEM_TEAM_INVALID, SHMEM_TEAM_NUM_CONTEXTS, &out);
    CHECK(0 != rc, "get_config(SHMEM_TEAM_INVALID) rc=0 expected nonzero");

    shmem_barrier_all();
    if (SHMEM_TEAM_INVALID != a) {
        shmem_team_destroy(a);
    }
    if (SHMEM_TEAM_INVALID != b) {
        shmem_team_destroy(b);
    }
    shmem_finalize();

    return nfail ? 1 : 0;
}
