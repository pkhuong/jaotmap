#include "interface.h"

#include <stdint.h>

static NO_INLINE __m256i
widget(__m256i x0, __m256i x1, __m256i neg_x, __m256i y0, __m256i neg_y)
{
        __m256i mask_x = neg_x ^ (x0 | x1);
        __m256i mask_y = neg_y ^ y0;

        return mask_x & mask_y;
}


static NO_INLINE __m256i
full_widget(__m256i noise, __m256i *dst, const __m256i *x0,
            const __m256i *x1, const __m256i *neg_x,
            const __m256i *y0, const __m256i *neg_y)
{
        __m256i ret;
        __m256i mask_x = *neg_x ^ (*x0 | *x1);
        __m256i mask_y = *neg_y ^ *y0;

        (void)noise;

        *dst = mask_x & mask_y;

        asm volatile("" : "=x"(ret));
        return ret;
}

void
specialised_widget(struct filter_state *restrict state)
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

        for (size_t i = 0; i < count; i++) {
                dst[i] = widget(x0[i], x1[i], neg_x[i], y0[i], neg_y[i]);
        }

        return;
}

void
fully_specialised_widget(struct filter_state *restrict state)
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

        for (size_t i = 0; i < count; i++) {
                __m256i noise;

                asm volatile("" : "=x"(noise));
                full_widget(noise, &dst[i], &x0[i], &x1[i], &neg_x[i],
                            &y0[i], &neg_y[i]);
        }

        return;
}
