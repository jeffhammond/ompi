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
 * Test shmem_put_signal and shmem_put_signal_nbi (default and context
 * forms, typed and putmem) with SHMEM_SIGNAL_SET and SHMEM_SIGNAL_ADD.
 * Each PE sends to its right neighbor; the receiver waits on the signal and
 * then checks that the data is already complete (data-then-signal
 * ordering), and finally checks the signal value with shmem_signal_fetch.
 *
 * NP: 4
 */

#include <stdio.h>
#include <stdint.h>
#include <shmem.h>

#define N 64
#define ROUNDS 5

static int failures = 0;

static int data[ROUNDS][N];
static int src[ROUNDS][N];
static uint64_t sig;

static void fail(int me, const char *what, long long expected, long long actual)
{
    printf("FAIL: PE %d %s expected %lld actual %lld\n", me, what, expected, actual);
    failures++;
}

static void check_row(int me, const char *what, int row, int from)
{
    int i;
    for (i = 0; i < N; i++) {
        int e = from * 1000 + row * 100 + i;
        if (data[row][i] != e) {
            fail(me, what, e, data[row][i]);
            return;
        }
    }
}

static void reset(void)
{
    int r, i;
    shmem_barrier_all();
    sig = 0;
    for (r = 0; r < ROUNDS; r++) {
        for (i = 0; i < N; i++) {
            data[r][i] = -1;
        }
    }
    shmem_barrier_all();
}

static void fill(int me)
{
    int r, i;
    for (r = 0; r < ROUNDS; r++) {
        for (i = 0; i < N; i++) {
            src[r][i] = me * 1000 + r * 100 + i;
        }
    }
}

/* variant: 0 put_signal, 1 put_signal_nbi, 2 ctx put_signal,
 * 3 ctx put_signal_nbi, 4 putmem_signal, 5 putmem_signal_nbi */
static void send(int variant, int r, int sig_op, uint64_t val, int peer)
{
    switch (variant) {
    case 0:
        shmem_int_put_signal(data[r], src[r], N, &sig, val, sig_op, peer);
        break;
    case 1:
        shmem_int_put_signal_nbi(data[r], src[r], N, &sig, val, sig_op, peer);
        break;
    case 2:
        shmem_ctx_int_put_signal(SHMEM_CTX_DEFAULT, data[r], src[r], N, &sig,
                                 val, sig_op, peer);
        break;
    case 3:
        shmem_ctx_int_put_signal_nbi(SHMEM_CTX_DEFAULT, data[r], src[r], N,
                                     &sig, val, sig_op, peer);
        break;
    case 4:
        shmem_putmem_signal(data[r], src[r], N * sizeof(int), &sig, val,
                            sig_op, peer);
        break;
    default:
        shmem_putmem_signal_nbi(data[r], src[r], N * sizeof(int), &sig, val,
                                sig_op, peer);
        break;
    }
}

static const char *names[] = {
    "put_signal", "put_signal_nbi", "ctx_put_signal", "ctx_put_signal_nbi",
    "putmem_signal", "putmem_signal_nbi"
};

int main(void)
{
    int me, npes, peer, left, v, r;
    char what[96];

    shmem_init();
    me = shmem_my_pe();
    npes = shmem_n_pes();
    if (npes < 2) {
        shmem_finalize();
        return 77;
    }
    peer = (me + 1) % npes;
    left = (me + npes - 1) % npes;
    fill(me);

    for (v = 0; v < 6; v++) {
        /* SIGNAL_SET: one put, signal becomes a distinctive value */
        reset();
        send(v, 0, SHMEM_SIGNAL_SET, UINT64_C(0x100000007), peer);
        shmem_uint64_wait_until(&sig, SHMEM_CMP_EQ, UINT64_C(0x100000007));
        snprintf(what, sizeof(what), "%s SET data", names[v]);
        check_row(me, what, 0, left);
        if (shmem_signal_fetch(&sig) != UINT64_C(0x100000007)) {
            snprintf(what, sizeof(what), "%s SET signal", names[v]);
            fail(me, what, (long long) UINT64_C(0x100000007),
                 (long long) shmem_signal_fetch(&sig));
        }
        shmem_quiet();

        /* SIGNAL_ADD: ROUNDS puts, each adds r+1; row r must be complete
         * once the cumulative signal reaches the sum through round r. */
        reset();
        for (r = 0; r < ROUNDS; r++) {
            send(v, r, SHMEM_SIGNAL_ADD, (uint64_t) (r + 1), peer);
        }
        {
            uint64_t cum = 0;
            for (r = 0; r < ROUNDS; r++) {
                cum += (uint64_t) (r + 1);
                shmem_uint64_wait_until(&sig, SHMEM_CMP_GE, cum);
                snprintf(what, sizeof(what), "%s ADD data row %d", names[v], r);
                check_row(me, what, r, left);
            }
            shmem_quiet();
            shmem_barrier_all();
            if (shmem_signal_fetch(&sig) != cum) {
                snprintf(what, sizeof(what), "%s ADD final signal", names[v]);
                fail(me, what, (long long) cum, (long long) shmem_signal_fetch(&sig));
            }
        }
    }

    shmem_barrier_all();
    shmem_finalize();
    if (0 != failures) {
        printf("FAIL: PE %d: %d failures\n", me, failures);
        return 1;
    }
    return 0;
}
