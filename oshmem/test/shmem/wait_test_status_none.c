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
 * Checks status-array exclusion and the results for an empty eligible set
 * of shmem_<type>_{test,wait_until}_{all,any,some} and the _vector forms:
 * test_all is true, any returns SIZE_MAX, some returns 0 and leaves the
 * indices array alone.  Also covers nelems == 0.
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <shmem.h>

#define NELEMS 8

static int me;
static int failures = 0;

#define FAIL(...)                                    \
    do {                                             \
        printf("FAIL: PE %d: ", me);                 \
        printf(__VA_ARGS__);                         \
        printf("\n");                                \
        failures++;                                  \
    } while (0)

#define EXPECT_SZ(what, got, want)                                             \
    do {                                                                       \
        size_t g_ = (got);                                                     \
        size_t w_ = (want);                                                    \
        if (g_ != w_) {                                                        \
            FAIL("%s " #what " (%s): returned %zu, expected %zu", tn, vname,   \
                 g_, w_);                                                      \
        }                                                                      \
    } while (0)

/* Run the checks once for one type and for the scalar or vector form. */
#define DEFINE_STATUS_TEST(N, T)                                               \
static T data_##N[NELEMS];                                                     \
static void status_##N(const char *tn, int vec)                                \
{                                                                              \
    const char *vname = vec ? "vector" : "scalar";                             \
    T val[NELEMS];                                                             \
    int st[NELEMS], i, ok;                                                     \
    size_t idx[NELEMS], r, k, want[NELEMS];                                    \
    for (i = 0; i < NELEMS; ++i) { data_##N[i] = 5; val[i] = 5; }              \
    /* a: everything eligible but everything excluded */                       \
    for (i = 0; i < NELEMS; ++i) { st[i] = 1; idx[i] = SIZE_MAX; }             \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, NELEMS, st,               \
                                           SHMEM_CMP_EQ, val)                  \
             : shmem_##N##_test_all(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);    \
    if (!ok) FAIL("%s test_all (%s) all excluded: returned 0", tn, vname);     \
    r = vec ? shmem_##N##_test_any_vector(data_##N, NELEMS, st,                \
                                          SHMEM_CMP_EQ, val)                   \
            : shmem_##N##_test_any(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);     \
    EXPECT_SZ(test_any all excluded, r, SIZE_MAX);                             \
    r = vec ? shmem_##N##_test_some_vector(data_##N, NELEMS, idx, st,          \
                                           SHMEM_CMP_EQ, val)                  \
            : shmem_##N##_test_some(data_##N, NELEMS, idx, st,                 \
                                    SHMEM_CMP_EQ, 5);                          \
    EXPECT_SZ(test_some all excluded, r, 0);                                   \
    for (k = 0; k < NELEMS; ++k) {                                             \
        if (idx[k] != SIZE_MAX) {                                              \
            FAIL("%s test_some (%s): indices[%zu] written when none eligible", \
                 tn, vname, k);                                                \
        }                                                                      \
    }                                                                          \
    if (vec) {                                                                 \
        shmem_##N##_wait_until_all_vector(data_##N, NELEMS, st,                \
                                          SHMEM_CMP_EQ, val);                  \
        r = shmem_##N##_wait_until_any_vector(data_##N, NELEMS, st,            \
                                              SHMEM_CMP_EQ, val);              \
        EXPECT_SZ(wait_until_any all excluded, r, SIZE_MAX);                   \
        r = shmem_##N##_wait_until_some_vector(data_##N, NELEMS, idx, st,      \
                                               SHMEM_CMP_EQ, val);             \
    } else {                                                                   \
        shmem_##N##_wait_until_all(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);     \
        r = shmem_##N##_wait_until_any(data_##N, NELEMS, st,                   \
                                       SHMEM_CMP_EQ, 5);                       \
        EXPECT_SZ(wait_until_any all excluded, r, SIZE_MAX);                   \
        r = shmem_##N##_wait_until_some(data_##N, NELEMS, idx, st,             \
                                        SHMEM_CMP_EQ, 5);                      \
    }                                                                          \
    EXPECT_SZ(wait_until_some all excluded, r, 0);                             \
    /* b: elements 0..2 eligible but excluded, 3..7 not eligible */            \
    for (i = 0; i < NELEMS; ++i) {                                             \
        st[i] = (i < 3);                                                       \
        data_##N[i] = (T) (i < 3 ? 5 : 6);                                     \
    }                                                                          \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, NELEMS, st,               \
                                           SHMEM_CMP_EQ, val)                  \
             : shmem_##N##_test_all(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);    \
    if (ok) FAIL("%s test_all (%s): true although 3..7 not eligible", tn,      \
                 vname);                                                       \
    r = vec ? shmem_##N##_test_any_vector(data_##N, NELEMS, st,                \
                                          SHMEM_CMP_EQ, val)                   \
            : shmem_##N##_test_any(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);     \
    EXPECT_SZ(test_any excluded only, r, SIZE_MAX);                            \
    /* c: element 5 eligible, 0..2 eligible but excluded */                    \
    data_##N[5] = 5;                                                           \
    r = vec ? shmem_##N##_test_any_vector(data_##N, NELEMS, st,                \
                                          SHMEM_CMP_EQ, val)                   \
            : shmem_##N##_test_any(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);     \
    EXPECT_SZ(test_any single eligible, r, 5);                                 \
    r = vec ? shmem_##N##_wait_until_any_vector(data_##N, NELEMS, st,          \
                                                SHMEM_CMP_EQ, val)             \
            : shmem_##N##_wait_until_any(data_##N, NELEMS, st,                 \
                                         SHMEM_CMP_EQ, 5);                     \
    EXPECT_SZ(wait_until_any single eligible, r, 5);                           \
    idx[0] = SIZE_MAX;                                                         \
    r = vec ? shmem_##N##_test_some_vector(data_##N, NELEMS, idx, st,          \
                                           SHMEM_CMP_EQ, val)                  \
            : shmem_##N##_test_some(data_##N, NELEMS, idx, st,                 \
                                    SHMEM_CMP_EQ, 5);                          \
    EXPECT_SZ(test_some single eligible, r, 1);                                \
    EXPECT_SZ(test_some single eligible index, idx[0], 5);                     \
    /* d: all but 0..2 eligible: exclusion lets test_all succeed */            \
    for (i = 0; i < NELEMS; ++i) data_##N[i] = (T) (i < 3 ? 6 : 5);            \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, NELEMS, st,               \
                                           SHMEM_CMP_EQ, val)                  \
             : shmem_##N##_test_all(data_##N, NELEMS, st, SHMEM_CMP_EQ, 5);    \
    if (!ok) FAIL("%s test_all (%s): false with ineligible elements "          \
                  "excluded", tn, vname);                                      \
    for (i = 0; i < NELEMS; ++i) { idx[i] = SIZE_MAX; want[i] = 0; }           \
    r = vec ? shmem_##N##_test_some_vector(data_##N, NELEMS, idx, st,          \
                                           SHMEM_CMP_EQ, val)                  \
            : shmem_##N##_test_some(data_##N, NELEMS, idx, st,                 \
                                    SHMEM_CMP_EQ, 5);                          \
    EXPECT_SZ(test_some five eligible, r, 5);                                  \
    for (k = 0; k < r && k < NELEMS; ++k) {                                    \
        if (idx[k] < 3 || idx[k] >= NELEMS || want[idx[k]]++) {                \
            FAIL("%s test_some (%s): bad index %zu", tn, vname, idx[k]);       \
        }                                                                      \
    }                                                                          \
    /* e: not eligible, not excluded */                                        \
    for (i = 0; i < NELEMS; ++i) { st[i] = 0; data_##N[i] = 6; }               \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, NELEMS, NULL,             \
                                           SHMEM_CMP_EQ, val)                  \
             : shmem_##N##_test_all(data_##N, NELEMS, NULL, SHMEM_CMP_EQ, 5);  \
    if (ok) FAIL("%s test_all (%s): true with nothing eligible", tn, vname);   \
    r = vec ? shmem_##N##_test_any_vector(data_##N, NELEMS, NULL,              \
                                          SHMEM_CMP_EQ, val)                   \
            : shmem_##N##_test_any(data_##N, NELEMS, NULL, SHMEM_CMP_EQ, 5);   \
    EXPECT_SZ(test_any nothing eligible, r, SIZE_MAX);                         \
    r = vec ? shmem_##N##_test_some_vector(data_##N, NELEMS, idx, NULL,        \
                                           SHMEM_CMP_EQ, val)                  \
            : shmem_##N##_test_some(data_##N, NELEMS, idx, NULL,               \
                                    SHMEM_CMP_EQ, 5);                          \
    EXPECT_SZ(test_some nothing eligible, r, 0);                               \
    /* f: nelems == 0 */                                                       \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, 0, NULL,                  \
                                           SHMEM_CMP_EQ, val)                  \
             : shmem_##N##_test_all(data_##N, 0, NULL, SHMEM_CMP_EQ, 5);       \
    if (!ok) FAIL("%s test_all (%s): false for nelems == 0", tn, vname);       \
    r = vec ? shmem_##N##_test_any_vector(data_##N, 0, NULL,                   \
                                          SHMEM_CMP_EQ, val)                   \
            : shmem_##N##_test_any(data_##N, 0, NULL, SHMEM_CMP_EQ, 5);        \
    EXPECT_SZ(test_any nelems 0, r, SIZE_MAX);                                 \
    r = vec ? shmem_##N##_test_some_vector(data_##N, 0, idx, NULL,             \
                                           SHMEM_CMP_EQ, val)                  \
            : shmem_##N##_test_some(data_##N, 0, idx, NULL, SHMEM_CMP_EQ, 5);  \
    EXPECT_SZ(test_some nelems 0, r, 0);                                       \
    if (vec) {                                                                 \
        shmem_##N##_wait_until_all_vector(data_##N, 0, NULL,                   \
                                          SHMEM_CMP_EQ, val);                  \
        r = shmem_##N##_wait_until_any_vector(data_##N, 0, NULL,               \
                                              SHMEM_CMP_EQ, val);              \
        EXPECT_SZ(wait_until_any nelems 0, r, SIZE_MAX);                       \
        r = shmem_##N##_wait_until_some_vector(data_##N, 0, idx, NULL,         \
                                               SHMEM_CMP_EQ, val);             \
    } else {                                                                   \
        shmem_##N##_wait_until_all(data_##N, 0, NULL, SHMEM_CMP_EQ, 5);        \
        r = shmem_##N##_wait_until_any(data_##N, 0, NULL, SHMEM_CMP_EQ, 5);    \
        EXPECT_SZ(wait_until_any nelems 0, r, SIZE_MAX);                       \
        r = shmem_##N##_wait_until_some(data_##N, 0, idx, NULL,                \
                                        SHMEM_CMP_EQ, 5);                      \
    }                                                                          \
    EXPECT_SZ(wait_until_some nelems 0, r, 0);                                 \
}

DEFINE_STATUS_TEST(short, short)
DEFINE_STATUS_TEST(int, int)
DEFINE_STATUS_TEST(long, long)
DEFINE_STATUS_TEST(uint64, uint64_t)
DEFINE_STATUS_TEST(size, size_t)

int main(void)
{
    int vec;

    shmem_init();
    me = shmem_my_pe();

    for (vec = 0; vec < 2; ++vec) {
        status_short("short", vec);
        status_int("int", vec);
        status_long("long", vec);
        status_uint64("uint64", vec);
        status_size("size", vec);
    }

    shmem_barrier_all();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failure(s)\n", me, failures);
    }
    shmem_finalize();
    return failures ? 1 : 0;
}
