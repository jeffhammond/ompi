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
 * Repeated calls to shmem_<type>_{test,wait_until}_any (and _vector) must
 * make progress across indices: when each returned index is added to the
 * status array, every eligible element is reported exactly once and the
 * final call reports none.  The second part has every other PE set one
 * flag on PE 0 in reverse order, and PE 0 collects them with repeated
 * wait_until_any calls.
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <shmem.h>

#define NELEMS 8

static int me, npes;
static int failures = 0;

#define FAIL(...)                                    \
    do {                                             \
        printf("FAIL: PE %d: ", me);                 \
        printf(__VA_ARGS__);                         \
        printf("\n");                                \
        failures++;                                  \
    } while (0)

#define DEFINE_PROGRESS_TEST(N, T)                                             \
static T local_##N[NELEMS];                                                    \
static void progress_##N(const char *tn)                                       \
{                                                                              \
    T val[NELEMS];                                                             \
    int st[NELEMS], seen[NELEMS], i, call, mode;                               \
    size_t r;                                                                  \
    for (mode = 0; mode < 4; ++mode) {                                         \
        /* 0,1: test_any scalar/vector; 2,3: wait_until_any scalar/vector */   \
        for (i = 0; i < NELEMS; ++i) {                                         \
            local_##N[i] = (T) (i % 2 ? 3 : 4);                                \
            val[i] = (T) (i % 2 ? 3 : 4);                                      \
            st[i] = 0;                                                         \
            seen[i] = 0;                                                       \
        }                                                                      \
        for (call = 0; call < NELEMS; ++call) {                                \
            switch (mode) {                                                    \
            case 0:                                                            \
                r = shmem_##N##_test_any(local_##N, NELEMS, st,                \
                                         SHMEM_CMP_GT, 2);                     \
                break;                                                         \
            case 1:                                                            \
                r = shmem_##N##_test_any_vector(local_##N, NELEMS, st,         \
                                                SHMEM_CMP_EQ, val);            \
                break;                                                         \
            case 2:                                                            \
                r = shmem_##N##_wait_until_any(local_##N, NELEMS, st,          \
                                               SHMEM_CMP_GT, 2);               \
                break;                                                         \
            default:                                                           \
                r = shmem_##N##_wait_until_any_vector(local_##N, NELEMS, st,   \
                                                      SHMEM_CMP_EQ, val);      \
                break;                                                         \
            }                                                                  \
            if (r >= NELEMS || seen[r]) {                                      \
                FAIL("%s any mode %d call %d: returned %zu", tn, mode, call,   \
                     r);                                                       \
                break;                                                         \
            }                                                                  \
            seen[r] = 1;                                                       \
            st[r] = 1;                                                         \
        }                                                                      \
        r = (mode < 2 ? (mode == 0 ?                                           \
                shmem_##N##_test_any(local_##N, NELEMS, st, SHMEM_CMP_GT, 2) : \
                shmem_##N##_test_any_vector(local_##N, NELEMS, st,             \
                                            SHMEM_CMP_EQ, val)) :              \
                (mode == 2 ?                                                   \
                shmem_##N##_wait_until_any(local_##N, NELEMS, st,              \
                                           SHMEM_CMP_GT, 2) :                  \
                shmem_##N##_wait_until_any_vector(local_##N, NELEMS, st,       \
                                                  SHMEM_CMP_EQ, val)));        \
        if (SIZE_MAX != r) {                                                   \
            FAIL("%s any mode %d: final call returned %zu, expected "          \
                 "SIZE_MAX", tn, mode, r);                                     \
        }                                                                      \
    }                                                                          \
}                                                                              \
static T *flags_##N;                                                           \
static void collect_##N(const char *tn)                                        \
{                                                                              \
    int st[NELEMS], seen[NELEMS], i, k;                                        \
    size_t r;                                                                  \
    flags_##N = (T *) shmem_malloc(NELEMS * sizeof(T));                        \
    if (NULL == flags_##N) { FAIL("%s: shmem_malloc failed", tn); return; }    \
    for (i = 0; i < NELEMS; ++i) flags_##N[i] = 0;                             \
    shmem_barrier_all();                                                       \
    if (0 == me) {                                                             \
        for (i = 0; i < NELEMS; ++i) { st[i] = (i < 1 || i >= npes); seen[i] = 0; } \
        for (k = 1; k < npes; ++k) {                                           \
            r = shmem_##N##_wait_until_any(flags_##N, NELEMS, st,              \
                                           SHMEM_CMP_NE, 0);                   \
            if (r < 1 || r >= (size_t) npes || seen[r]) {                      \
                FAIL("%s collect: call %d returned %zu", tn, k, r);            \
                break;                                                         \
            }                                                                  \
            seen[r] = 1;                                                       \
            st[r] = 1;                                                         \
        }                                                                      \
        r = shmem_##N##_wait_until_any(flags_##N, NELEMS, st, SHMEM_CMP_NE, 0);\
        if (SIZE_MAX != r) {                                                   \
            FAIL("%s collect: final call returned %zu", tn, r);                \
        }                                                                      \
    } else {                                                                   \
        usleep(15000 * (npes - me));                                           \
        shmem_##N##_p(&flags_##N[me], (T) 1, 0);                               \
        shmem_quiet();                                                         \
    }                                                                          \
    shmem_barrier_all();                                                       \
    shmem_free(flags_##N);                                                     \
}

DEFINE_PROGRESS_TEST(short, short)
DEFINE_PROGRESS_TEST(int, int)
DEFINE_PROGRESS_TEST(long, long)
DEFINE_PROGRESS_TEST(uint64, uint64_t)
DEFINE_PROGRESS_TEST(size, size_t)

int main(void)
{
    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2 || npes > NELEMS) {
        shmem_finalize();
        return 77;
    }

    progress_short("short");
    progress_int("int");
    progress_long("long");
    progress_uint64("uint64");
    progress_size("size");

    collect_short("short");
    collect_int("int");
    collect_long("long");
    collect_uint64("uint64");
    collect_size("size");

    shmem_barrier_all();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failure(s)\n", me, failures);
    }
    shmem_finalize();
    return failures ? 1 : 0;
}
