#include "interface.h"

void
baseline(struct filter_state *restrict state)
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
                __m256i mask_x = neg_x[i] ^ (x0[i] | x1[i]);
                __m256i mask_y = neg_y[i] ^ y0[i];
                dst[i] = mask_x & mask_y;
        }

        return;
}
