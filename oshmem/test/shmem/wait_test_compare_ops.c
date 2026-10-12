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
 * Checks shmem_<type>_{test,wait_until}_{all,any,some} and their _vector
 * forms for all six comparison operators on locally initialized data.
 * Blocking forms are only called with data that already satisfies them.
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

static const int ops[6] = {
    SHMEM_CMP_EQ, SHMEM_CMP_NE, SHMEM_CMP_GT,
    SHMEM_CMP_GE, SHMEM_CMP_LT, SHMEM_CMP_LE
};
static const char *opnames[6] = { "EQ", "NE", "GT", "GE", "LT", "LE" };

static int ref(long long a, int cmp, long long b)
{
    switch (cmp) {
    case SHMEM_CMP_EQ: return a == b;
    case SHMEM_CMP_NE: return a != b;
    case SHMEM_CMP_GT: return a > b;
    case SHMEM_CMP_GE: return a >= b;
    case SHMEM_CMP_LT: return a < b;
    default:           return a <= b;
    }
}

static void check_indices(const char *what, const char *tn, const char *opn,
                          int vec, size_t n, const size_t *idx,
                          const int *elig, int expect_n)
{
    int seen[NELEMS] = { 0 };
    size_t k;

    if (n != (size_t) expect_n) {
        FAIL("%s %s %s%s: returned %zu, expected %d", what, tn, opn,
             vec ? " vector" : "", n, expect_n);
        return;
    }
    for (k = 0; k < n; ++k) {
        if (idx[k] >= NELEMS || !elig[idx[k]] || seen[idx[k]]) {
            FAIL("%s %s %s%s: bad index %zu at position %zu", what, tn, opn,
                 vec ? " vector" : "", idx[k], k);
            return;
        }
        seen[idx[k]] = 1;
    }
}

#define DEFINE_OPS_TEST(N, T)                                                  \
static T data_##N[NELEMS];                                                     \
static void ops_##N(int oi, int vec)                                           \
{                                                                              \
    T val[NELEMS];                                                             \
    T scalar = 14;                                                             \
    int elig[NELEMS], mask[NELEMS], n = 0, i, ok;                              \
    size_t idx[NELEMS], r, k;                                                  \
    const char *opn = opnames[oi];                                             \
    int op = ops[oi];                                                          \
    for (i = 0; i < NELEMS; ++i) {                                             \
        data_##N[i] = (T) (10 + i);                                            \
        val[i] = (T) ((i % 2 == 0) ? 10 + i : 9 + i);                          \
        elig[i] = ref((long long) data_##N[i], op,                             \
                      (long long) (vec ? val[i] : scalar));                    \
        mask[i] = !elig[i];                                                    \
        n += elig[i];                                                          \
    }                                                                          \
    ok = vec ? shmem_##N##_test_all_vector(data_##N, NELEMS, NULL, op, val)    \
             : shmem_##N##_test_all(data_##N, NELEMS, NULL, op, scalar);       \
    if ((ok != 0) != (n == NELEMS)) {                                          \
        FAIL("test_all " #N " %s%s: returned %d, expected %d", opn,            \
             vec ? " vector" : "", ok, n == NELEMS);                           \
    }                                                                          \
    r = vec ? shmem_##N##_test_any_vector(data_##N, NELEMS, NULL, op, val)     \
            : shmem_##N##_test_any(data_##N, NELEMS, NULL, op, scalar);        \
    if (0 == n ? (r != SIZE_MAX) : (r >= NELEMS || !elig[r])) {                \
        FAIL("test_any " #N " %s%s: returned %zu with %d eligible", opn,       \
             vec ? " vector" : "", r, n);                                      \
    }                                                                          \
    for (k = 0; k < NELEMS; ++k) idx[k] = SIZE_MAX;                            \
    r = vec ? shmem_##N##_test_some_vector(data_##N, NELEMS, idx, NULL, op,    \
                                           val)                                \
            : shmem_##N##_test_some(data_##N, NELEMS, idx, NULL, op, scalar);  \
    check_indices("test_some", #N, opn, vec, r, idx, elig, n);                 \
    /* Blocking forms: exclude every element that is not eligible. */          \
    if (vec) {                                                                 \
        shmem_##N##_wait_until_all_vector(data_##N, NELEMS, mask, op, val);    \
    } else {                                                                   \
        shmem_##N##_wait_until_all(data_##N, NELEMS, mask, op, scalar);        \
    }                                                                          \
    if (n > 0) {                                                               \
        r = vec ? shmem_##N##_wait_until_any_vector(data_##N, NELEMS, NULL,    \
                                                    op, val)                   \
                : shmem_##N##_wait_until_any(data_##N, NELEMS, NULL, op,       \
                                             scalar);                          \
        if (r >= NELEMS || !elig[r]) {                                         \
            FAIL("wait_until_any " #N " %s%s: returned %zu", opn,              \
                 vec ? " vector" : "", r);                                     \
        }                                                                      \
        for (k = 0; k < NELEMS; ++k) idx[k] = SIZE_MAX;                        \
        r = vec ? shmem_##N##_wait_until_some_vector(data_##N, NELEMS, idx,    \
                                                     NULL, op, val)            \
                : shmem_##N##_wait_until_some(data_##N, NELEMS, idx, NULL,     \
                                              op, scalar);                     \
        check_indices("wait_until_some", #N, opn, vec, r, idx, elig, n);       \
    }                                                                          \
}

DEFINE_OPS_TEST(short, short)
DEFINE_OPS_TEST(int, int)
DEFINE_OPS_TEST(long, long)
DEFINE_OPS_TEST(uint64, uint64_t)
DEFINE_OPS_TEST(size, size_t)

int main(void)
{
    int oi, vec;

    shmem_init();
    me = shmem_my_pe();

    for (oi = 0; oi < 6; ++oi) {
        for (vec = 0; vec < 2; ++vec) {
            ops_short(oi, vec);
            ops_int(oi, vec);
            ops_long(oi, vec);
            ops_uint64(oi, vec);
            ops_size(oi, vec);
        }
    }

    shmem_barrier_all();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failure(s)\n", me, failures);
    }
    shmem_finalize();
    return failures ? 1 : 0;
}
