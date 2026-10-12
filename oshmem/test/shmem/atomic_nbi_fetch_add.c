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
 * NP: 8
 * Nonblocking fetch-add AMOs, default and context forms.
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

#define FOREACH_INT_TYPE(X)       \
    X(int, int)                   \
    X(long, long)                 \
    X(long long, longlong)        \
    X(unsigned int, uint)         \
    X(unsigned long, ulong)       \
    X(unsigned long long, ulonglong) \
    X(int32_t, int32)             \
    X(int64_t, int64)             \
    X(uint32_t, uint32)           \
    X(uint64_t, uint64)           \
    X(size_t, size)               \
    X(ptrdiff_t, ptrdiff)

#define FADD_K 4

/*
 * Every PE issues FADD_K nonblocking fetch-adds of 1 to one counter on
 * PE 0.  After quiet, the fetched values over all PEs must be exactly
 * 0..npes*FADD_K-1, and the counter must hold npes*FADD_K.  A second
 * counter takes PE-dependent addends and must hold their sum.
 */
#define DEFINE_FADD(T, TN)                                                  \
static void test_fadd_##TN(int form)                                        \
{                                                                           \
    T *tgt = (T *) sym_tgt;                                                 \
    T fetch[FADD_K], fetch2[FADD_K], final1, final2;                        \
    uint64_t mine[FADD_K], all[MAXPES * FADD_K];                            \
    int total = npes * FADD_K, seen[MAXPES * FADD_K], i;                    \
    uint64_t sum = 0;                                                       \
                                                                            \
    tgt[0] = 0;                                                             \
    tgt[1] = 0;                                                             \
    for (i = 0; i < FADD_K; ++i) {                                          \
        fetch[i] = SENTINEL(T);                                             \
        fetch2[i] = SENTINEL(T);                                            \
    }                                                                       \
    shmem_barrier_all();                                                    \
    for (i = 0; i < FADD_K; ++i) {                                          \
        NBI(form, TN, fetch_add, &fetch[i], &tgt[0], (T) 1, 0);             \
        NBI(form, TN, fetch_add, &fetch2[i], &tgt[1], (T) (me + 1), 0);     \
    }                                                                       \
    QUIET(form);                                                            \
    for (i = 0; i < FADD_K; ++i) {                                          \
        mine[i] = (uint64_t) fetch[i];                                      \
        CHECK(fetch2[i] != SENTINEL(T), form, #TN, "fetch2 unwritten",      \
              0, (uint64_t) fetch2[i]);                                     \
    }                                                                       \
    shmem_putmem(&sym_gather[me * FADD_K], mine, sizeof(mine), 0);          \
    shmem_barrier_all();                                                    \
    shmem_getmem(all, sym_gather, total * sizeof(uint64_t), 0);             \
    for (i = 0; i < total; ++i) {                                           \
        seen[i] = 0;                                                        \
    }                                                                       \
    for (i = 0; i < total; ++i) {                                           \
        if (all[i] >= (uint64_t) total) {                                   \
            CHECK(0, form, #TN, "fetched value out of range", total - 1,    \
                  all[i]);                                                  \
        } else {                                                            \
            ++seen[all[i]];                                                 \
        }                                                                   \
    }                                                                       \
    for (i = 0; i < total; ++i) {                                           \
        CHECK(1 == seen[i], form, #TN, "fetched value multiplicity",        \
              1, seen[i]);                                                  \
    }                                                                       \
    final1 = shmem_##TN##_atomic_fetch(&tgt[0], 0);                         \
    final2 = shmem_##TN##_atomic_fetch(&tgt[1], 0);                         \
    for (i = 0; i < npes; ++i) {                                            \
        sum += (uint64_t) (i + 1) * FADD_K;                                 \
    }                                                                       \
    CHECK((uint64_t) final1 == (uint64_t) total, form, #TN, "final counter",\
          total, (uint64_t) final1);                                        \
    CHECK((uint64_t) final2 == sum, form, #TN, "final weighted counter",    \
          sum, (uint64_t) final2);                                          \
    shmem_barrier_all();                                                    \
}

FOREACH_INT_TYPE(DEFINE_FADD)

int main(void)
{
    int rc = setup(4);

    if (0 != rc) {
        return rc;
    }
#define RUN_FADD(T, TN)                         \
    FOREACH_FORM_BEGIN test_fadd_##TN(form); FOREACH_FORM_END
    FOREACH_INT_TYPE(RUN_FADD)
    return teardown("atomic_nbi_fetch_add");
}
