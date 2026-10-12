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
 * The deprecated active-set shmem_sync(PE_start, logPE_stride, PE_size,
 * pSync) keeps working (it returns void) next to the team-based
 * shmem_team_sync, with SHMEM_SYNC_VALUE initialized pSync arrays reused
 * across calls.
 *
 * NP: 4
 */

#include <stdio.h>
#include <shmem.h>

static int errors = 0;
static long psync_all[SHMEM_SYNC_SIZE];
static long psync_even[SHMEM_SYNC_SIZE];

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
    int me, npes, i, k;
    int *val;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 4) {
        shmem_finalize();
        return 77;
    }
    for (i = 0; i < SHMEM_SYNC_SIZE; i++) {
        psync_all[i] = SHMEM_SYNC_VALUE;
        psync_even[i] = SHMEM_SYNC_VALUE;
    }
    val = shmem_malloc(sizeof(int));
    if (NULL == val) {
        printf("FAIL: PE %d: shmem_malloc\n", me);
        shmem_finalize();
        return 1;
    }
    shmem_barrier_all();

    /* whole world, repeated, pSync reused */
    for (k = 1; k <= 10; k++) {
        *val = -1;
        shmem_sync(0, 0, npes, psync_all);
        shmem_int_p(val, 100 * k + me, (me + 1) % npes);
        shmem_sync(0, 0, npes, psync_all);
        CHECK(*val == 100 * k + (me + npes - 1) % npes, me,
              "value after shmem_sync", 100 * k + (me + npes - 1) % npes,
              *val);
    }

    /* even PEs only: start 0, log stride 1 */
    if (me % 2 == 0) {
        int members = (npes + 1) / 2;
        for (k = 1; k <= 10; k++) {
            int idx = me / 2;
            *val = -1;
            shmem_sync(0, 1, members, psync_even);
            shmem_int_p(val, 7 * k + idx, 2 * ((idx + 1) % members));
            shmem_sync(0, 1, members, psync_even);
            CHECK(*val == 7 * k + (idx + members - 1) % members, me,
                  "value after active-set sync",
                  7 * k + (idx + members - 1) % members, *val);
        }
    }

    /* mixing with the team API */
    CHECK(shmem_team_sync(SHMEM_TEAM_WORLD) == 0, me,
          "team sync after shmem_sync", 0, 1);
    shmem_sync(0, 0, npes, psync_all);
    shmem_barrier_all();
    shmem_free(val);
    shmem_finalize();
    return errors ? 1 : 0;
}
