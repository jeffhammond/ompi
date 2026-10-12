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
 * Team broadcast from every root on SHMEM_TEAM_WORLD, strided teams and
 * 2D-split teams.  The destination must be written on the root as well
 * (OpenSHMEM 1.5), and back-to-back calls must not interfere.
 *
 * NP: 8
 */

#include <shmem.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int nfail = 0;

#define FAIL(...)                                          \
    do {                                                   \
        if (nfail < 30) {                                  \
            printf("FAIL: PE %d: ", shmem_my_pe());        \
            printf(__VA_ARGS__);                           \
            printf("\n");                                  \
            fflush(stdout);                                \
        }                                                  \
        nfail++;                                           \
    } while (0)

/*
 * The team cases every test runs over.  All PEs create the same list in the
 * same order; a PE that is not a member of a case holds SHMEM_TEAM_INVALID.
 */
enum { NCASE = 5 };
static shmem_team_t tm[NCASE];
static const char *tname[NCASE] = { "world", "strided(1,2)", "strided(0,3)",
                                    "2d-x", "2d-y" };

static void make_teams(int n)
{
    int i, rc;
    int xrange = (n > 4) ? 3 : 2;

    for (i = 0; i < NCASE; i++) {
        tm[i] = SHMEM_TEAM_INVALID;
    }
    tm[0] = SHMEM_TEAM_WORLD;
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 1, 2, n / 2, NULL, 0,
                                  &tm[1]);
    if (0 != rc) {
        FAIL("split_strided(1,2) returned %d", rc);
    }
    rc = shmem_team_split_strided(SHMEM_TEAM_WORLD, 0, 3, (n + 2) / 3, NULL, 0,
                                  &tm[2]);
    if (0 != rc) {
        FAIL("split_strided(0,3) returned %d", rc);
    }
    rc = shmem_team_split_2d(SHMEM_TEAM_WORLD, xrange, NULL, 0, &tm[3], NULL, 0,
                             &tm[4]);
    if (0 != rc) {
        FAIL("split_2d(%d) returned %d", xrange, rc);
    }
}

static void destroy_teams(void)
{
    int i;

    for (i = 1; i < NCASE; i++) {
        shmem_team_destroy(tm[i]);
    }
}

enum { NB = 7, NK = 6 };

static void *sbuf, *dbuf;

static int gen(int root, int k, int rep)
{
    return (root * 17 + k * 3 + rep * 5 + 1) % 100;
}

/* Broadcast from each team rank in turn, with a barrier between roots. */
#define BCAST_ROOTS(TN, T)                                                    \
    do {                                                                      \
        T *s_ = (T *) sbuf, *d_ = (T *) dbuf;                                 \
        for (root = 0; root < tn_max; root++) {                               \
            shmem_sync_all();                                                 \
            for (k = 0; k < NB; k++) {                                        \
                s_[k] = (member && root < tn)                                 \
                        ? (T) gen(root, k, 0) : (T) 77;                       \
                d_[k] = (T) 101;                                              \
            }                                                                 \
            shmem_sync_all();                                                 \
            if (member && root < tn) {                                        \
                rc = shmem_##TN##_broadcast(team, d_, s_, NB, root);          \
                if (0 != rc) {                                                \
                    FAIL("%s %s_broadcast root %d returned %d", tname[c],     \
                         #TN, root, rc);                                      \
                }                                                             \
                for (k = 0; k < NB; k++) {                                    \
                    T e = (T) gen(root, k, 0);                                \
                    if (d_[k] != e) {                                         \
                        FAIL("%s %s_broadcast root %d (team rank %d) elem "   \
                             "%d: expected %d, got %d", tname[c], #TN, root,  \
                             tme, k, (int) e, (int) d_[k]);                   \
                    }                                                         \
                }                                                             \
            }                                                                 \
        }                                                                     \
    } while (0)

static void all_types(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    int tn_max = shmem_n_pes();
    int root, k, rc;

    BCAST_ROOTS(char, char);
    BCAST_ROOTS(short, short);
    BCAST_ROOTS(int, int);
    BCAST_ROOTS(long, long);
    BCAST_ROOTS(longlong, long long);
    BCAST_ROOTS(float, float);
    BCAST_ROOTS(double, double);
    BCAST_ROOTS(longdouble, long double);
    BCAST_ROOTS(uint8, uint8_t);
    BCAST_ROOTS(int32, int32_t);
    BCAST_ROOTS(uint64, uint64_t);
    BCAST_ROOTS(size, size_t);
    BCAST_ROOTS(ptrdiff, ptrdiff_t);
}

/* Untyped broadcast, in bytes. */
static void broadcastmem(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    unsigned char *s = (unsigned char *) sbuf, *d = (unsigned char *) dbuf;
    int root, k, rc;

    for (root = 0; root < shmem_n_pes(); root++) {
        shmem_sync_all();
        for (k = 0; k < 37; k++) {
            s[k] = (member && root < tn) ? (unsigned char) gen(root, k, 1)
                                         : 77;
            d[k] = 0xEE;
        }
        shmem_sync_all();
        if (member && root < tn) {
            rc = shmem_broadcastmem(team, d, s, 37, root);
            if (0 != rc) {
                FAIL("%s broadcastmem root %d returned %d", tname[c], root, rc);
            }
            for (k = 0; k < 37; k++) {
                if (d[k] != (unsigned char) gen(root, k, 1)) {
                    FAIL("%s broadcastmem root %d (team rank %d) byte %d: "
                         "expected %d, got %d", tname[c], root, tme, k,
                         gen(root, k, 1), (int) d[k]);
                }
            }
        }
    }
}

/* NK broadcasts in a row without synchronization, with rotating roots. */
static void back_to_back(int c)
{
    shmem_team_t team = tm[c];
    int member = (SHMEM_TEAM_INVALID != team);
    int tme = member ? shmem_team_my_pe(team) : -1;
    int tn = member ? shmem_team_n_pes(team) : 0;
    int *s = (int *) shmem_malloc(NK * NB * sizeof(int));
    int *d = (int *) shmem_malloc(NK * NB * sizeof(int));
    int j, k, rc;

    if (NULL == s || NULL == d) {
        FAIL("shmem_malloc failed");
        return;
    }
    for (j = 0; j < NK; j++) {
        for (k = 0; k < NB; k++) {
            s[j * NB + k] = member ? gen(j % (tn ? tn : 1), k, j) : 0;
            d[j * NB + k] = 101;
        }
    }
    shmem_sync_all();
    if (member) {
        for (j = 0; j < NK; j++) {
            rc = shmem_int_broadcast(team, d + j * NB, s + j * NB, NB, j % tn);
            if (0 != rc) {
                FAIL("%s back-to-back broadcast %d returned %d", tname[c], j,
                     rc);
            }
        }
        for (j = 0; j < NK; j++) {
            for (k = 0; k < NB; k++) {
                int e = gen(j % tn, k, j);
                if (d[j * NB + k] != e) {
                    FAIL("%s back-to-back broadcast %d (team rank %d) elem "
                         "%d: expected %d, got %d", tname[c], j, tme, k, e,
                         d[j * NB + k]);
                }
            }
        }
    }
    shmem_sync_all();
    shmem_free(d);
    shmem_free(s);
}

int main(void)
{
    int c, n, pe, failed;

    shmem_init();
    pe = shmem_my_pe();
    n = shmem_n_pes();
    if (n < 2) {
        if (0 == pe) {
            printf("SKIP: need at least 2 PEs\n");
        }
        shmem_finalize();
        return 77;
    }

    sbuf = shmem_malloc(NB * sizeof(long double) + 64);
    dbuf = shmem_malloc(NB * sizeof(long double) + 64);
    if (NULL == sbuf || NULL == dbuf) {
        FAIL("shmem_malloc failed");
        shmem_global_exit(1);
    }
    make_teams(n);

    for (c = 0; c < NCASE; c++) {
        all_types(c);
        broadcastmem(c);
        back_to_back(c);
    }

    shmem_sync_all();
    destroy_teams();
    shmem_free(dbuf);
    shmem_free(sbuf);
    failed = (0 != nfail);
    shmem_finalize();
    return failed ? 1 : 0;
}
