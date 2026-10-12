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
 * NP: 4
 *
 * Blocking shmem_<type>_wait_until_{all,any,some} and the _vector forms,
 * satisfied by shmem_p from the left neighbour PE after a short delay.
 * No atomics are used to set the remote values.
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <shmem.h>

#define NELEMS 4

static int me, npes, left, right;
static int failures = 0;

#define FAIL(...)                                    \
    do {                                             \
        printf("FAIL: PE %d: ", me);                 \
        printf(__VA_ARGS__);                         \
        printf("\n");                                \
        failures++;                                  \
    } while (0)

#define DEFINE_REMOTE_TEST(N, T)                                               \
static void remote_##N(const char *tn)                                         \
{                                                                              \
    T *iv = (T *) shmem_malloc(NELEMS * sizeof(T));                            \
    T vals[NELEMS];                                                            \
    int st[NELEMS], i, vec, round;                                             \
    size_t idx[NELEMS], r, got = 0;                                            \
    int seen[NELEMS];                                                          \
    if (NULL == iv) { FAIL("%s: shmem_malloc failed", tn); return; }           \
    for (i = 0; i < NELEMS; ++i) vals[i] = (T) (100 + 7 * i);                  \
    for (vec = 0; vec < 2; ++vec) {                                            \
        /* all: the neighbour fills every element */                           \
        for (i = 0; i < NELEMS; ++i) iv[i] = 0;                                \
        shmem_barrier_all();                                                   \
        usleep(20000);                                                         \
        for (i = 0; i < NELEMS; ++i) {                                         \
            shmem_##N##_p(&iv[i], vals[i], right);                             \
        }                                                                      \
        if (vec) {                                                             \
            shmem_##N##_wait_until_all_vector(iv, NELEMS, NULL,                \
                                              SHMEM_CMP_EQ, vals);             \
        } else {                                                               \
            shmem_##N##_wait_until_all(iv, NELEMS, NULL, SHMEM_CMP_GE, 100);   \
        }                                                                      \
        for (i = 0; i < NELEMS; ++i) {                                         \
            if (iv[i] != vals[i]) {                                            \
                FAIL("%s wait_until_all%s: iv[%d] = %lld, expected %lld", tn,  \
                     vec ? "_vector" : "", i, (long long) iv[i],               \
                     (long long) vals[i]);                                     \
            }                                                                  \
        }                                                                      \
        shmem_quiet();                                                         \
        /* any: the neighbour sets only element 2 */                           \
        for (i = 0; i < NELEMS; ++i) iv[i] = 0;                                \
        shmem_barrier_all();                                                   \
        usleep(20000);                                                         \
        shmem_##N##_p(&iv[2], vals[2], right);                                 \
        r = vec ? shmem_##N##_wait_until_any_vector(iv, NELEMS, NULL,          \
                                                    SHMEM_CMP_EQ, vals)        \
                : shmem_##N##_wait_until_any(iv, NELEMS, NULL, SHMEM_CMP_NE,   \
                                             0);                               \
        if (2 != r) {                                                          \
            FAIL("%s wait_until_any%s: returned %zu, expected 2", tn,          \
                 vec ? "_vector" : "", r);                                     \
        }                                                                      \
        shmem_quiet();                                                         \
        /* some: the neighbour sets elements 1 and 3, in two steps; each */    \
        /* call excludes what was already reported */                          \
        for (i = 0; i < NELEMS; ++i) { iv[i] = 0; st[i] = 0; seen[i] = 0; }    \
        st[0] = st[2] = 1;                                                     \
        shmem_barrier_all();                                                   \
        usleep(20000);                                                         \
        shmem_##N##_p(&iv[1], vals[1], right);                                 \
        usleep(20000);                                                         \
        shmem_##N##_p(&iv[3], vals[3], right);                                 \
        got = 0;                                                               \
        for (round = 0; round < 2 && got < 2; ++round) {                       \
            size_t k;                                                          \
            r = vec ? shmem_##N##_wait_until_some_vector(iv, NELEMS, idx, st,  \
                                                         SHMEM_CMP_EQ, vals)   \
                    : shmem_##N##_wait_until_some(iv, NELEMS, idx, st,         \
                                                  SHMEM_CMP_NE, 0);            \
            if (r < 1 || r > 2) {                                              \
                FAIL("%s wait_until_some%s: returned %zu", tn,                 \
                     vec ? "_vector" : "", r);                                 \
                break;                                                         \
            }                                                                  \
            for (k = 0; k < r; ++k) {                                          \
                if ((idx[k] != 1 && idx[k] != 3) || seen[idx[k]]) {            \
                    FAIL("%s wait_until_some%s: bad index %zu", tn,            \
                         vec ? "_vector" : "", idx[k]);                        \
                    break;                                                     \
                }                                                              \
                seen[idx[k]] = 1;                                              \
                st[idx[k]] = 1;                                                \
                got++;                                                         \
            }                                                                  \
        }                                                                      \
        if (got != 2) {                                                        \
            FAIL("%s wait_until_some%s: saw %zu of 2 elements", tn,            \
                 vec ? "_vector" : "", got);                                   \
        }                                                                      \
        shmem_quiet();                                                         \
    }                                                                          \
    shmem_barrier_all();                                                       \
    shmem_free(iv);                                                            \
}

DEFINE_REMOTE_TEST(short, short)
DEFINE_REMOTE_TEST(int, int)
DEFINE_REMOTE_TEST(long, long)
DEFINE_REMOTE_TEST(uint64, uint64_t)
DEFINE_REMOTE_TEST(size, size_t)

int main(void)
{
    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2) {
        shmem_finalize();
        return 77;
    }
    left = (me + npes - 1) % npes;
    right = (me + 1) % npes;
    (void) left;

    remote_short("short");
    remote_int("int");
    remote_long("long");
    remote_uint64("uint64");
    remote_size("size");

    shmem_barrier_all();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failure(s)\n", me, failures);
    }
    shmem_finalize();
    return failures ? 1 : 0;
}
