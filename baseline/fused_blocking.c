#include "interface.h"

static NO_INLINE __m256i
block_xor_or(__m256i noise,
    __m256i *restrict dst, const __m256i *restrict neg,
    const __m256i *restrict x, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                dst[i] = neg[i] ^ (x[i] | y[i]);

        asm volatile("" : "=x"(ret));
        return ret;
}

static NO_INLINE __m256i
nblock_and_xor(__m256i noise, __m256i *restrict acc,
    const __m256i *restrict x, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                acc[i] &= x[i] ^ y[i];

        asm volatile("" : "=x"(ret));
        return ret;
}

void
fused_blocking(struct filter_state *restrict state)
{
        size_t count = state->count;
        __m256i *restrict dst = state->dst;
        const __m256i *restrict x0 = state->x0;
        const __m256i *restrict x1 = state->x1;
        const __m256i *restrict neg_x = state->neg_x;
        const __m256i *restrict y0 = state->y0;
        const __m256i *restrict neg_y = state->neg_y;

        if ((count % BLOCK_SIZE) != 0)
                __builtin_unreachable();

        for (size_t i = 0; i < count; i+= BLOCK_SIZE) {
                __m256i noise;

                asm volatile("" : "=x"(noise));
                block_xor_or(noise, dst + i, neg_x + i, x0 + i, x1 + i);
                asm volatile("" : "=x"(noise));
                nblock_and_xor(noise, dst + i, neg_y + i, y0 + i);
        }

        return;
}
