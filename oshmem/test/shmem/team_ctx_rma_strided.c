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
 * RMA and AMO through a context created on a strided team.  The PE argument
 * of a context operation is team-relative: team PE i is world PE
 * start + i * stride.  Each team member targets team PE (i + 1) % size and
 * the value must land on the corresponding physical PE only.
 *
 * NP: 8
 */

#include <stdio.h>
#include <shmem.h>

#define START  1
#define STRIDE 3
#define TSIZE  3   /* world PEs 1, 4, 7 */

static long dst_p = -1;
static long dst_get = 0;
static long dst_amo = 100;
static long dst_fetch = -1;

int main(void)
{
    shmem_team_t team = SHMEM_TEAM_INVALID;
    shmem_ctx_t ctx = SHMEM_CTX_INVALID;
    int me, npes, rc, nfail = 0, tme, tn, i, in_team;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < START + (TSIZE - 1) * STRIDE + 1) {
        shmem_finalize();
        return 77;
    }

    dst_get = 1000 + me;   /* readable by neighbours */
    in_team = (me >= START && me <= START + (TSIZE - 1) * STRIDE
               && 0 == (me - START) % STRIDE);

    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, START, STRIDE, TSIZE, NULL, 0, &team);
    if (0 != rc) {
        printf("FAIL: PE %d: split_strided rc=%d\n", me, rc);
        nfail++;
    }
    shmem_barrier_all();

    if (in_team && SHMEM_TEAM_INVALID != team) {
        long got, old, v;
        int target;

        tme = shmem_team_my_pe(team);
        tn = shmem_team_n_pes(team);
        if (tme != (me - START) / STRIDE || tn != TSIZE) {
            printf("FAIL: PE %d: team pe/size %d/%d expected %d/%d\n",
                   me, tme, tn, (me - START) / STRIDE, TSIZE);
            nfail++;
        }
        rc = shmem_team_create_ctx(team, 0, &ctx);
        if (0 != rc || SHMEM_CTX_INVALID == ctx) {
            printf("FAIL: PE %d: create_ctx rc=%d\n", me, rc);
            nfail++;
        } else {
            target = (tme + 1) % TSIZE;
            shmem_ctx_long_p(ctx, &dst_p, 500 + me, target);
            shmem_ctx_quiet(ctx);

            /* get from team PE target: expect its world-PE-specific value */
            got = shmem_ctx_long_g(ctx, &dst_get, target);
            i = START + target * STRIDE;
            if (got != 1000 + i) {
                printf("FAIL: PE %d: ctx_g from team PE %d (world %d) got %ld expected %d\n",
                       me, target, i, got, 1000 + i);
                nfail++;
            }

            /* AMO to a different target: team PE (tme + 2) % TSIZE */
            target = (tme + 2) % TSIZE;
            v = 1;
            shmem_ctx_long_atomic_add(ctx, &dst_amo, 10 + tme, target);
            old = shmem_ctx_long_atomic_fetch_add(ctx, &dst_fetch, v, target);
            (void) old;
            shmem_ctx_quiet(ctx);
        }
    }

    shmem_barrier_all();

    /* verify the put landed on the right physical PE */
    {
        long expect = -1;
        if (in_team) {
            int pred_t = (shmem_team_my_pe(team) + TSIZE - 1) % TSIZE;
            expect = 500 + START + pred_t * STRIDE;
        }
        if (dst_p != expect) {
            printf("FAIL: PE %d: ctx_p result %ld expected %ld\n", me, dst_p, expect);
            nfail++;
        }
    }

    /* verify the AMO: sender is team PE (tme - 2) mod TSIZE */
    {
        long expect = 100;
        long fexp = -1;
        if (in_team) {
            int snd = (shmem_team_my_pe(team) + TSIZE - 2) % TSIZE;
            expect = 100 + 10 + snd;
            fexp = -1 + 1;
        }
        if (dst_amo != expect) {
            printf("FAIL: PE %d: ctx_atomic_add result %ld expected %ld\n", me, dst_amo, expect);
            nfail++;
        }
        if (dst_fetch != fexp) {
            printf("FAIL: PE %d: ctx_atomic_fetch_add result %ld expected %ld\n", me, dst_fetch, fexp);
            nfail++;
        }
    }

    if (SHMEM_CTX_INVALID != ctx) {
        shmem_ctx_destroy(ctx);
    }
    shmem_barrier_all();
    if (SHMEM_TEAM_INVALID != team) {
        shmem_team_destroy(team);
    }
    shmem_finalize();

    return nfail ? 1 : 0;
}
