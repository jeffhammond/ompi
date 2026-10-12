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
 * Create and destroy many teams (strided, nested, 2D) in a loop and check
 * that the symmetric heap stays symmetric: shmem_malloc must return the
 * same offset on every PE afterwards, and teams must remain functional.
 * Non-members of a split must not allocate symmetric memory.
 *
 * NP: 4
 */

#include <stdio.h>
#include <stdint.h>
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

#define NLOOP 200

static int check_symmetry(int me, int npes, const char *when)
{
    long addr_in;
    int i, ok = 1;
    void *p = shmem_malloc(1024);

    if (NULL == p) {
        printf("FAIL: PE %d: shmem_malloc failed %s\n", me, when);
        return 0;
    }
    /* compare the address with every other PE's through a symmetric int */
    {
        long *sym = shmem_malloc(sizeof(long));
        long *all = shmem_malloc(npes * sizeof(long));
        addr_in = (long) (intptr_t) p;
        *sym = addr_in;
        shmem_barrier_all();
        for (i = 0; i < npes; i++) {
            all[i] = shmem_long_g(sym, i);
        }
        for (i = 0; i < npes; i++) {
            if (all[i] != addr_in) {
                printf("FAIL: PE %d: heap not symmetric %s: PE %d returned "
                       "%#lx, I got %#lx\n", me, when, i,
                       (unsigned long) all[i], (unsigned long) addr_in);
                ok = 0;
            }
        }
        shmem_barrier_all();
        shmem_free(all);
        shmem_free(sym);
    }
    shmem_free(p);
    return ok;
}

int main(void)
{
    int me, npes, it, rc;
    int *buf;

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();

    if (npes < 4) {
        shmem_finalize();
        return 77;
    }

    if (!check_symmetry(me, npes, "before loop")) {
        errors++;
    }

    buf = shmem_malloc(sizeof(int));
    for (it = 0; it < NLOOP; it++) {
        shmem_team_t t = SHMEM_TEAM_INVALID, c = SHMEM_TEAM_INVALID;
        int start = it % 2, stride = 1 + (it % 3);
        int size = (npes - start + stride - 1) / stride;
        int member = (me >= start) && ((me - start) % stride == 0);

        rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, size,
                                      NULL, 0, &t);
        CHECK(member == (t != SHMEM_TEAM_INVALID), me, "validity", member,
              t != SHMEM_TEAM_INVALID);
        if (t != SHMEM_TEAM_INVALID) {
            CHECK(0 == rc, me, "split return", 0, rc);
            CHECK(shmem_team_sync(t) == 0, me, "sync return", 0, 1);
            if (size > 1) {
                /* nested team of all but the first member */
                shmem_team_split_strided(t, 1, 1, size - 1, NULL, 0, &c);
                if (c != SHMEM_TEAM_INVALID) {
                    CHECK(shmem_team_sync(c) == 0, me, "nested sync", 0, 1);
                }
                shmem_team_destroy(c);
            }
            shmem_team_sync(t);
        }
        shmem_team_destroy(t);

        if (it % 10 == 0) {
            shmem_team_t x = SHMEM_TEAM_INVALID, y = SHMEM_TEAM_INVALID;
            rc = shmem_team_split_2d(SHMEM_TEAM_WORLD, 3, NULL, 0, &x, NULL,
                                     0, &y);
            CHECK(0 == rc, me, "2d split return", 0, rc);
            if (x != SHMEM_TEAM_INVALID) shmem_team_sync(x);
            if (y != SHMEM_TEAM_INVALID) shmem_team_sync(y);
            shmem_team_destroy(y);
            shmem_team_destroy(x);
        }

        /* an allocation in the middle of the loop must stay symmetric */
        if (it % 50 == 49) {
            if (!check_symmetry(me, npes, "inside loop")) {
                errors++;
            }
        }
    }

    shmem_barrier_all();
    if (!check_symmetry(me, npes, "after loop")) {
        errors++;
    }

    /* the world team still works */
    *buf = -1;
    shmem_team_sync(SHMEM_TEAM_WORLD);
    shmem_int_p(buf, me, (me + 1) % npes);
    rc = shmem_team_sync(SHMEM_TEAM_WORLD);
    CHECK(0 == rc, me, "final sync return", 0, rc);
    CHECK(*buf == (me + npes - 1) % npes, me, "final ring value",
          (me + npes - 1) % npes, *buf);

    shmem_free(buf);
    shmem_finalize();
    return errors ? 1 : 0;
}
