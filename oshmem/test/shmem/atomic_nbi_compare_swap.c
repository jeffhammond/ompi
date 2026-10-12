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
 * Nonblocking compare-swap AMOs, default and context forms.
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

/*
 * Ring phase: a compare-swap with a matching condition must fetch the old
 * value and install the new one; one with a non-matching condition must
 * fetch the current value and leave the target alone.  Many-to-one phase:
 * every PE tries to swap 0 -> me+1 on PE 0; exactly one succeeds.
 */
#define DEFINE_CSWAP(T, TN)                                                 \
static void test_cswap_##TN(int form)                                       \
{                                                                           \
    T *tgt = (T *) sym_tgt;                                                 \
    T f_ok, f_bad, f_one, fin;                                              \
    uint64_t mine, all[MAXPES];                                             \
    int nb = (me + 1) % npes, prev = (me + npes - 1) % npes, i;             \
    int winners = 0;                                                        \
                                                                            \
    tgt[0] = (T) (50 + me);                                                 \
    f_ok = SENTINEL(T);                                                     \
    f_bad = SENTINEL(T);                                                    \
    shmem_barrier_all();                                                    \
    NBI(form, TN, compare_swap, &f_ok, &tgt[0], (T) (50 + nb),              \
        (T) (70 + me), nb);                                                 \
    QUIET(form);                                                            \
    shmem_barrier_all();                                                    \
    CHECK(f_ok == (T) (50 + nb), form, #TN, "success fetched",              \
          50 + nb, (uint64_t) f_ok);                                        \
    CHECK(tgt[0] == (T) (70 + prev), form, #TN, "success final",            \
          70 + prev, (uint64_t) tgt[0]);                                    \
    shmem_barrier_all();                                                    \
    NBI(form, TN, compare_swap, &f_bad, &tgt[0], (T) 999, (T) 1, nb);       \
    QUIET(form);                                                            \
    shmem_barrier_all();                                                    \
    CHECK(f_bad == (T) (70 + me), form, #TN, "failure fetched",             \
          70 + me, (uint64_t) f_bad);                                       \
    CHECK(tgt[0] == (T) (70 + prev), form, #TN, "failure final",            \
          70 + prev, (uint64_t) tgt[0]);                                    \
                                                                            \
    tgt[1] = 0;                                                             \
    f_one = SENTINEL(T);                                                    \
    shmem_barrier_all();                                                    \
    NBI(form, TN, compare_swap, &f_one, &tgt[1], (T) 0, (T) (me + 1), 0);   \
    QUIET(form);                                                            \
    mine = (uint64_t) f_one;                                                \
    shmem_putmem(&sym_gather[me], &mine, sizeof(mine), 0);                  \
    shmem_barrier_all();                                                    \
    shmem_getmem(all, sym_gather, npes * sizeof(uint64_t), 0);              \
    fin = shmem_##TN##_atomic_fetch(&tgt[1], 0);                            \
    CHECK((uint64_t) fin >= 1 && (uint64_t) fin <= (uint64_t) npes, form,   \
          #TN, "winner value in 1..npes", npes, (uint64_t) fin);            \
    for (i = 0; i < npes; ++i) {                                            \
        if (0 == all[i]) {                                                  \
            ++winners;                                                      \
            CHECK((uint64_t) fin == (uint64_t) (i + 1), form, #TN,          \
                  "winner's value is final", i + 1, (uint64_t) fin);        \
        } else {                                                            \
            CHECK(all[i] == (uint64_t) fin, form, #TN,                      \
                  "loser fetched winner's value", (uint64_t) fin, all[i]);  \
        }                                                                   \
    }                                                                       \
    CHECK(1 == winners, form, #TN, "number of winners", 1, winners);        \
    shmem_barrier_all();                                                    \
}

FOREACH_INT_TYPE(DEFINE_CSWAP)

int main(void)
{
    int rc = setup(4);

    if (0 != rc) {
        return rc;
    }
#define RUN_CSWAP(T, TN)                        \
    FOREACH_FORM_BEGIN test_cswap_##TN(form); FOREACH_FORM_END
    FOREACH_INT_TYPE(RUN_CSWAP)
    return teardown("atomic_nbi_compare_swap");
}
