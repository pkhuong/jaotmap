#pragma once

#include <immintrin.h>

#define NO_INLINE __attribute__((noinline, noclone))

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

struct filter_state {
        /* Count in __m256i; must be a multiple of 32. */
        size_t count;
        union {
                struct {
                        __m256i *dst;
                        const __m256i *x0;
                        const __m256i *x1;
                        const __m256i *neg_x;

                        const __m256i *y0;
                        const __m256i *neg_y;
                };
                __m256i *ptrs[6];
        };

        struct {
                size_t index;
                __m256i val[4 * BLOCK_SIZE];
        } scratch;
};

void noop(struct filter_state *);

/**
 * The loop as I'd write it in normal code.
 */
void baseline(struct filter_state *);

/**
 * Small (L1-sized) blocks for each individual operation.
 *
 * ~numpy style.
 */
void blocking(struct filter_state *);

/**
 * Add some fused operators to the above.
 *
 *  - (xor neg (or x y))
 *  - (and acc (xor neg x))
 */
void fused_blocking(struct filter_state *);

/**
 * What if we had a widget for one iteration of that loop?
 */
void specialised_widget(struct filter_state *);

void fully_specialised_widget(struct filter_state *);

/**
 * Simulate inline threading while passing values in registers.
 */
void threaded_inreg(struct filter_state *);

/**
 * Add some fused operators to the above.
 */
void threaded_inreg_fused(struct filter_state *);

/**
 * Hardwire the "next" calls.
 */
void wired_inreg_fused(struct filter_state *);
