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
 * C11 _Generic dispatch with const-qualified pointers.  shmem_g,
 * shmem_atomic_fetch and shmem_atomic_fetch_nbi are called with const and
 * non-const pointers (with and without a context) for every type in the
 * generic tables, and the vector wait_until/test routines are called with
 * a const comparison array.  Results are verified at run time.
 */

#include <stdio.h>
#include <stdlib.h>
#include <shmem.h>

#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L

int main(void)
{
    printf("SKIP: C11 is required\n");
    return 77;
}

#else

#define NELEMS 4

static int fails = 0;

/*
 * The nonblocking fetch and the vector wait_until/test calls are only
 * compiled, not executed, by default: the UCX SPML does not implement the
 * vector routines and rejects nonblocking fetches into local variables.
 * Set OSHMEM_TEST_ALL_APIS to execute them as well.
 */
static int run_nbi = 0;
static int run_vec = 0;

#define EXPECT(what, got, exp)                                            \
    do {                                                                  \
        long double got_ = (long double) (got);                           \
        long double exp_ = (long double) (exp);                           \
        if (got_ != exp_) {                                               \
            printf("FAIL: PE %d: %s: expected %Lg, got %Lg\n", me, what,  \
                   exp_, got_);                                           \
            fails++;                                                      \
        }                                                                 \
    } while (0)

/* Types handled by shmem_g: X(name, type) */
#define G_TYPES(X)                                                        \
    X(char, char) X(short, short) X(int, int) X(long, long)               \
    X(longlong, long long) X(schar, signed char)                          \
    X(uchar, unsigned char) X(ushort, unsigned short)                     \
    X(uint, unsigned int) X(ulong, unsigned long)                         \
    X(ulonglong, unsigned long long) X(float, float) X(double, double)    \
    X(longdouble, long double)

/* Types handled by shmem_atomic_fetch and shmem_atomic_fetch_nbi */
#define F_TYPES(X)                                                        \
    X(int, int) X(long, long) X(longlong, long long)                      \
    X(uint, unsigned int) X(ulong, unsigned long)                         \
    X(ulonglong, unsigned long long) X(float, float) X(double, double)

/* Types handled by the vector wait_until/test routines */
#define V_TYPES(X)                                                        \
    X(short, short) X(ushort, unsigned short) X(int, int) X(long, long)   \
    X(longlong, long long) X(uint, unsigned int) X(ulong, unsigned long)  \
    X(ulonglong, unsigned long long)

#define DECL_G(n, t) static t sym_g_##n;
#define DECL_F(n, t) static t sym_f_##n;
#define DECL_V(n, t) static t sym_v_##n[NELEMS];
G_TYPES(DECL_G)
F_TYPES(DECL_F)
V_TYPES(DECL_V)

#define DEF_G(n, t)                                                       \
static void run_g_##n(int me, int peer)                                   \
{                                                                         \
    const t *cp = &sym_g_##n;                                             \
    t *const mp = &sym_g_##n;                                             \
    const t *const cpc = &sym_g_##n;                                      \
    t exp = (t) (peer + 1);                                               \
    EXPECT("shmem_g(const " #t " *)", shmem_g(cp, peer), exp);            \
    EXPECT("shmem_g(" #t " *)", shmem_g(mp, peer), exp);                  \
    EXPECT("shmem_g(const " #t " *const)", shmem_g(cpc, peer), exp);      \
    EXPECT("shmem_g(ctx, const " #t " *)",                                \
           shmem_g(SHMEM_CTX_DEFAULT, cp, peer), exp);                    \
    EXPECT("shmem_g(ctx, " #t " *)",                                      \
           shmem_g(SHMEM_CTX_DEFAULT, mp, peer), exp);                    \
}
G_TYPES(DEF_G)

#define DEF_F(n, t)                                                       \
static void run_f_##n(int me, int peer)                                   \
{                                                                         \
    const t *cp = &sym_f_##n;                                             \
    t *mp = &sym_f_##n;                                                   \
    t exp = (t) (peer + 1);                                               \
    t r1, r2, r3, r4;                                                     \
    EXPECT("shmem_atomic_fetch(const " #t " *)",                          \
           shmem_atomic_fetch(cp, peer), exp);                            \
    EXPECT("shmem_atomic_fetch(" #t " *)",                                \
           shmem_atomic_fetch(mp, peer), exp);                            \
    EXPECT("shmem_atomic_fetch(ctx, const " #t " *)",                     \
           shmem_atomic_fetch(SHMEM_CTX_DEFAULT, cp, peer), exp);         \
    EXPECT("shmem_atomic_fetch(ctx, " #t " *)",                           \
           shmem_atomic_fetch(SHMEM_CTX_DEFAULT, mp, peer), exp);         \
    /* The nonblocking fetch is compile-checked only (see run_nbi). */   \
    r1 = r2 = r3 = r4 = 0;                                                \
    if (run_nbi) {                                                        \
        shmem_atomic_fetch_nbi(&r1, cp, peer);                            \
        shmem_atomic_fetch_nbi(&r2, mp, peer);                            \
        shmem_atomic_fetch_nbi(SHMEM_CTX_DEFAULT, &r3, cp, peer);         \
        shmem_atomic_fetch_nbi(SHMEM_CTX_DEFAULT, &r4, mp, peer);         \
        shmem_quiet();                                                    \
        EXPECT("shmem_atomic_fetch_nbi(const " #t " *)", r1, exp);        \
        EXPECT("shmem_atomic_fetch_nbi(" #t " *)", r2, exp);              \
        EXPECT("shmem_atomic_fetch_nbi(ctx, const " #t " *)", r3, exp);   \
        EXPECT("shmem_atomic_fetch_nbi(ctx, " #t " *)", r4, exp);         \
    }                                                                     \
}
F_TYPES(DEF_F)

/*
 * ivars = {1, 2, 3, 4}.  With the comparison array {9, 9, 3, 9} and
 * SHMEM_CMP_EQ only element 2 is satisfied; with {1, 2, 3, 4} all are.
 */
#define DEF_V(n, t)                                                       \
static void run_v_##n(int me)                                             \
{                                                                         \
    static const t all[NELEMS] = { 1, 2, 3, 4 };                          \
    static const t one[NELEMS] = { 9, 9, 3, 9 };                          \
    const int status[NELEMS] = { 0, 0, 0, 0 };                            \
    size_t idx[NELEMS] = { 0, 0, 0, 0 };                                  \
    const t *cmp_all = all;                                               \
    const t *cmp_one = one;                                               \
    t *iv = sym_v_##n;                                                    \
    if (!run_vec) {                                                       \
        return;                                                           \
    }                                                                     \
    shmem_wait_until_all_vector(iv, NELEMS, status, SHMEM_CMP_EQ,         \
                                cmp_all);                                 \
    EXPECT("wait_until_any_vector(" #t ")",                               \
           shmem_wait_until_any_vector(iv, NELEMS, status, SHMEM_CMP_EQ,  \
                                       cmp_one), 2);                      \
    EXPECT("wait_until_some_vector(" #t ") count",                        \
           shmem_wait_until_some_vector(iv, NELEMS, idx, status,          \
                                        SHMEM_CMP_EQ, cmp_one), 1);       \
    EXPECT("wait_until_some_vector(" #t ") index", idx[0], 2);            \
    EXPECT("test_all_vector(" #t ") match",                               \
           shmem_test_all_vector(iv, NELEMS, status, SHMEM_CMP_EQ,        \
                                 cmp_all), 1);                            \
    EXPECT("test_all_vector(" #t ") mismatch",                            \
           shmem_test_all_vector(iv, NELEMS, status, SHMEM_CMP_EQ,        \
                                 cmp_one), 0);                            \
    EXPECT("test_any_vector(" #t ")",                                     \
           shmem_test_any_vector(iv, NELEMS, status, SHMEM_CMP_EQ,        \
                                 cmp_one), 2);                            \
    idx[0] = 0;                                                           \
    EXPECT("test_some_vector(" #t ") count",                              \
           shmem_test_some_vector(iv, NELEMS, idx, status,                \
                                  SHMEM_CMP_EQ, cmp_one), 1);             \
    EXPECT("test_some_vector(" #t ") index", idx[0], 2);                  \
}
V_TYPES(DEF_V)

#define INIT_G(n, t) sym_g_##n = (t) (me + 1);
#define INIT_F(n, t) sym_f_##n = (t) (me + 1);
#define INIT_V(n, t) { int i; for (i = 0; i < NELEMS; i++) sym_v_##n[i] = (t) (i + 1); }
#define CALL_G(n, t) run_g_##n(me, peer);
#define CALL_F(n, t) run_f_##n(me, peer);
#define CALL_V(n, t) run_v_##n(me);

int main(void)
{
    int me, npes, peer;

    run_nbi = run_vec = (NULL != getenv("OSHMEM_TEST_ALL_APIS"));
    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    peer = (me + 1) % npes;

    G_TYPES(INIT_G)
    F_TYPES(INIT_F)
    V_TYPES(INIT_V)
    shmem_barrier_all();

    G_TYPES(CALL_G)
    F_TYPES(CALL_F)
    V_TYPES(CALL_V)

    shmem_barrier_all();
    shmem_finalize();
    return fails ? 1 : 0;
}

#endif
