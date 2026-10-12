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
 * Nonblocking fetch-and/or/xor AMOs, default and context forms.
 */

#include <shmem.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define MAXPES 16
#define GATHER_SLOTS 64

static int me, npes, nfail;
static shmem_ctx_t created_ctx;
static uint64_t *sym_gather;
static void *sym_tgt;

/* Forms: 0 = no context, 1 = SHMEM_CTX_DEFAULT, 2 = a created context. */
#define NFORMS 3

static const char *form_name[NFORMS] = { "default", "ctx_default", "ctx_created" };

static shmem_ctx_t form_ctx(int form)
{
    return (1 == form) ? SHMEM_CTX_DEFAULT : created_ctx;
}

#define QUIET(form)                         \
    do {                                    \
        if (0 == (form)) {                  \
            shmem_quiet();                  \
        } else {                            \
            shmem_ctx_quiet(form_ctx(form)); \
        }                                   \
    } while (0)

#define NBI(form, TN, OP, ...)                                              \
    do {                                                                    \
        if (0 == (form)) {                                                  \
            shmem_##TN##_atomic_##OP##_nbi(__VA_ARGS__);                    \
        } else {                                                            \
            shmem_ctx_##TN##_atomic_##OP##_nbi(form_ctx(form), __VA_ARGS__); \
        }                                                                   \
    } while (0)

#define CHECK(cond, form, TN, what, expected, actual)                       \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL: PE %d: %s [%s]: %s: expected %llu, actual %llu\n", \
                   me, TN, form_name[form], what,                           \
                   (unsigned long long) (expected),                         \
                   (unsigned long long) (actual));                          \
            ++nfail;                                                        \
        }                                                                   \
    } while (0)

#define FOREACH_FORM_BEGIN  { int form; for (form = 0; form < NFORMS; ++form) {
#define FOREACH_FORM_END    } }

static int setup(int required_pes)
{
    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < required_pes || npes > MAXPES) {
        if (0 == me) {
            printf("SKIP: need between %d and %d PEs, have %d\n",
                   required_pes, MAXPES, npes);
        }
        shmem_finalize();
        return 77;
    }
    sym_gather = shmem_malloc(GATHER_SLOTS * sizeof(uint64_t));
    sym_tgt = shmem_malloc(8 * sizeof(uint64_t));
    if (NULL == sym_gather || NULL == sym_tgt) {
        printf("FAIL: PE %d: shmem_malloc failed\n", me);
        shmem_global_exit(1);
    }
    if (0 != shmem_ctx_create(0, &created_ctx)) {
        /* Contexts unavailable: exercise the default context instead. */
        created_ctx = SHMEM_CTX_DEFAULT;
    }
    shmem_barrier_all();
    return 0;
}

static int teardown(const char *name)
{
    shmem_barrier_all();
    if (SHMEM_CTX_DEFAULT != created_ctx) {
        shmem_ctx_destroy(created_ctx);
    }
    shmem_free(sym_gather);
    shmem_free(sym_tgt);
    if (nfail) {
        printf("FAIL: %s: PE %d saw %d failure(s)\n", name, me, nfail);
    }
    shmem_finalize();
    return nfail ? 1 : 0;
}

/* Fetched values start as all-ones/-1 so an unwritten buffer is detected. */
#define SENTINEL(T) ((T) -1)

#define FOREACH_BIT_TYPE(X)       \
    X(unsigned int, uint)         \
    X(unsigned long, ulong)       \
    X(unsigned long long, ulonglong) \
    X(int32_t, int32)             \
    X(int64_t, int64)             \
    X(uint32_t, uint32)           \
    X(uint64_t, uint64)

#define BIT_K 4

#define INIT_VAL(T, p, j) \
    ((T) (0xF0F0F0F0A5A5A5A5ULL ^ ((uint64_t) (p) * 0x0101010101ULL) ^ \
          ((uint64_t) (j) * 0x1111ULL)))
#define OPND_VAL(T, p, j) \
    ((T) (0x3C3C5A5A0FF0F00FULL + (uint64_t) (p) * 7 + (uint64_t) (j) * 0x10001ULL))
#define PE_BIT(T, p) ((T) ((uint64_t) 1 << (8 * sizeof(T) - 1 - (p))))

#define OP_and(a, b) ((a) & (b))
#define OP_or(a, b)  ((a) | (b))
#define OP_xor(a, b) ((a) ^ (b))

/*
 * Ring phase: each PE applies BIT_K nonblocking ops to its right
 * neighbor, which is the only writer, so both the fetched (old) value and
 * the final value are fully determined.  Many-to-one phase: every PE
 * touches its own bit of one word on PE 0.
 */
#define DEFINE_BITOP(T, TN, OP, ALLSET_INIT)                                \
static void test_##OP##_##TN(int form)                                      \
{                                                                           \
    volatile T *tgt = (volatile T *) sym_tgt;                               \
    T fetch[BIT_K], mask = 0, bit, f1, expect, one_fetch;                   \
    int nb = (me + 1) % npes, prev = (me + npes - 1) % npes, i;             \
                                                                            \
    for (i = 0; i < BIT_K; ++i) {                                           \
        tgt[i] = INIT_VAL(T, me, i);                                        \
        fetch[i] = SENTINEL(T);                                             \
    }                                                                       \
    shmem_barrier_all();                                                    \
    for (i = 0; i < BIT_K; ++i) {                                           \
        NBI(form, TN, fetch_##OP, &fetch[i], (T *) &tgt[i],                 \
            OPND_VAL(T, me, i), nb);                                        \
    }                                                                       \
    QUIET(form);                                                            \
    shmem_barrier_all();                                                    \
    for (i = 0; i < BIT_K; ++i) {                                           \
        CHECK(fetch[i] == INIT_VAL(T, nb, i), form, #TN,                    \
              #OP " ring fetched", (uint64_t) INIT_VAL(T, nb, i),           \
              (uint64_t) fetch[i]);                                         \
        expect = OP_##OP(INIT_VAL(T, me, i), OPND_VAL(T, prev, i));         \
        CHECK(tgt[i] == expect, form, #TN, #OP " ring final",               \
              (uint64_t) expect, (uint64_t) tgt[i]);                        \
    }                                                                       \
    for (i = 0; i < npes; ++i) {                                            \
        mask |= PE_BIT(T, i);                                               \
    }                                                                       \
    tgt[0] = (T) (ALLSET_INIT);                                             \
    one_fetch = SENTINEL(T);                                                \
    shmem_barrier_all();                                                    \
    bit = PE_BIT(T, me);                                                    \
    NBI(form, TN, fetch_##OP, &one_fetch, (T *) &tgt[0],                    \
        (0 == strcmp(#OP, "and")) ? (T) ~bit : bit, 0);                     \
    QUIET(form);                                                            \
    shmem_barrier_all();                                                    \
    f1 = shmem_##TN##_atomic_fetch((T *) &tgt[0], 0);                       \
    expect = (0 == strcmp(#OP, "and")) ? (T) ~mask : mask;                  \
    CHECK(f1 == expect, form, #TN, #OP " many-to-one final",                \
          (uint64_t) expect, (uint64_t) f1);                                \
    if (0 == strcmp(#OP, "and")) {                                          \
        CHECK(one_fetch & bit, form, #TN, "and many-to-one fetched bit",    \
              (uint64_t) bit, (uint64_t) (one_fetch & bit));                \
    } else {                                                                \
        CHECK(!(one_fetch & bit), form, #TN,                                \
              #OP " many-to-one fetched bit", 0,                            \
              (uint64_t) (one_fetch & bit));                                \
    }                                                                       \
    shmem_barrier_all();                                                    \
}

#define DEFINE_ALL_OPS(T, TN)                   \
    DEFINE_BITOP(T, TN, and, (T) ~(T) 0)        \
    DEFINE_BITOP(T, TN, or, (T) 0)              \
    DEFINE_BITOP(T, TN, xor, (T) 0)

FOREACH_BIT_TYPE(DEFINE_ALL_OPS)

int main(void)
{
    int rc = setup(4);

    if (0 != rc) {
        return rc;
    }
#define RUN_ALL_OPS(T, TN)                                                  \
    FOREACH_FORM_BEGIN                                                      \
        test_and_##TN(form);                                                \
        test_or_##TN(form);                                                 \
        test_xor_##TN(form);                                                \
    FOREACH_FORM_END
    FOREACH_BIT_TYPE(RUN_ALL_OPS)
    return teardown("atomic_nbi_bitops");
}
