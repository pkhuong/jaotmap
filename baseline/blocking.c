#include "interface.h"

/**
 * Return an AVX 256 value to avoid VZEROUPPER.
 */
static NO_INLINE __m256i
block_or(__m256i noise, __m256i *restrict dst,
    const __m256i *restrict x, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                dst[i] = x[i] | y[i];

        asm volatile("" : "=x"(ret));
        return ret;
}

static NO_INLINE __m256i
block_and(__m256i noise, __m256i *restrict dst,
    const __m256i *restrict x, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                dst[i] = x[i] & y[i];

        asm volatile("" : "=x"(ret));
        return ret;
}

static NO_INLINE __m256i
block_xor(__m256i noise, __m256i *restrict dst,
    const __m256i *restrict x, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                dst[i] = x[i] ^ y[i];

        asm volatile("" : "=x"(ret));
        return ret;
}

static NO_INLINE __m256i
nblock_xor(__m256i noise, __m256i *restrict dst, const __m256i *restrict y)
{
        __m256i ret;

        (void)noise;
        for (size_t i = 0; i < BLOCK_SIZE; i++)
                dst[i] ^= y[i];

        asm volatile("" : "=x"(ret));
        return ret;
}

void
blocking(struct filter_state *restrict state)
{
        size_t count = state->count;
        __m256i *restrict dst = state->dst;
        const __m256i *restrict x0 = state->x0;
        const __m256i *restrict x1 = state->x1;
        const __m256i *restrict neg_x = state->neg_x;
        const __m256i *restrict y0 = state->y0;
        const __m256i *restrict neg_y = state->neg_y;
        __m256i *restrict tmp0 = &state->scratch.val[0];
        __m256i *restrict tmp1 = &state->scratch.val[BLOCK_SIZE];

        if ((count % BLOCK_SIZE) != 0)
                __builtin_unreachable();

        for (size_t i = 0; i < count; i+= BLOCK_SIZE) {
                __m256i noise;

                asm volatile("" : "=x"(noise));
                block_or(noise, tmp0, x0 + i, x1 + i);
                asm volatile("" : "=x"(noise));
                nblock_xor(noise, tmp0, neg_x + i);
                asm volatile("" : "=x"(noise));
                block_xor(noise, tmp1, y0 + i, neg_y + i);
                asm volatile("" : "=x"(noise));
                block_and(noise, dst + i, tmp0, tmp1);
        }

        return;
}
