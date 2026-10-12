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
 * Test shmem_atomic_set for all standard types, in both the default
 * and the context forms, targeting a remote PE.
 *
 * NP: 4
 */

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <shmem.h>

static int failures = 0;
static uint64_t var_big;

#define DECLARE(name, type) static type var_##name; static type var_ctx_##name;

#define VAL(type, v) ((type) (v))

DECLARE(int, int)
DECLARE(long, long)
DECLARE(longlong, long long)
DECLARE(uint, unsigned int)
DECLARE(ulong, unsigned long)
DECLARE(ulonglong, unsigned long long)
DECLARE(float, float)
DECLARE(double, double)
DECLARE(int32, int32_t)
DECLARE(int64, int64_t)
DECLARE(uint32, uint32_t)
DECLARE(uint64, uint64_t)
DECLARE(size, size_t)
DECLARE(ptrdiff, ptrdiff_t)

/* Each PE writes a PE-specific value into its right neighbor. */
#define SET(name, type, me, peer)                                       \
    do {                                                                \
        shmem_##name##_atomic_set(&var_##name, VAL(type, 100 + (me)), (peer)); \
        shmem_ctx_##name##_atomic_set(SHMEM_CTX_DEFAULT, &var_ctx_##name, \
                                      VAL(type, 200 + (me)), (peer));   \
    } while (0)

#define CHECK(name, type, me, left)                                     \
    do {                                                                \
        type e1 = VAL(type, 100 + (left)), e2 = VAL(type, 200 + (left)); \
        if (var_##name != e1) {                                         \
            printf("FAIL: PE %d shmem_" #name "_atomic_set expected %g actual %g\n", \
                   (me), (double) e1, (double) var_##name);             \
            failures++;                                                 \
        }                                                               \
        if (var_ctx_##name != e2) {                                     \
            printf("FAIL: PE %d shmem_ctx_" #name "_atomic_set expected %g actual %g\n", \
                   (me), (double) e2, (double) var_ctx_##name);         \
            failures++;                                                 \
        }                                                               \
    } while (0)

#define FOR_ALL(M, ...)                         \
    M(int, int, __VA_ARGS__);                    \
    M(long, long, __VA_ARGS__);                  \
    M(longlong, long long, __VA_ARGS__);         \
    M(uint, unsigned int, __VA_ARGS__);          \
    M(ulong, unsigned long, __VA_ARGS__);        \
    M(ulonglong, unsigned long long, __VA_ARGS__); \
    M(float, float, __VA_ARGS__);                \
    M(double, double, __VA_ARGS__);              \
    M(int32, int32_t, __VA_ARGS__);              \
    M(int64, int64_t, __VA_ARGS__);              \
    M(uint32, uint32_t, __VA_ARGS__);            \
    M(uint64, uint64_t, __VA_ARGS__);            \
    M(size, size_t, __VA_ARGS__);                \
    M(ptrdiff, ptrdiff_t, __VA_ARGS__);

int main(void)
{
    int me, npes, peer, left;
    int64_t big_expected = INT64_C(0x123456789abcdef0);

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2) {
        shmem_finalize();
        return 77;
    }
    peer = (me + 1) % npes;
    left = (me + npes - 1) % npes;

    shmem_barrier_all();

    FOR_ALL(SET, me, peer)
    /* 64-bit value that does not fit in 32 bits */
    shmem_uint64_atomic_set(&var_big, (uint64_t) big_expected, peer);

    shmem_barrier_all();

    FOR_ALL(CHECK, me, left)
    if (var_big != (uint64_t) big_expected) {
        printf("FAIL: PE %d shmem_uint64_atomic_set expected 0x%llx actual 0x%llx\n",
               me, (unsigned long long) big_expected,
               (unsigned long long) var_big);
        failures++;
    }

    shmem_barrier_all();
    shmem_finalize();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failures\n", me, failures);
        return 1;
    }
    return 0;
}
