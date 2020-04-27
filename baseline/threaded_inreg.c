#include "interface.h"

#include <stdint.h>

struct op_list;

/**
 * We'll do inline threading opcodes. SysV allows 6 scalar arguments,
 * and 8 SSE.
 *
 * We have the filter state, the list of operation, our index in that
 * list, and the loop iteration index * 32.
 *
 * ip indicates the next instruction * sizeof(struct op).
 */
typedef void op_t(struct filter_state *, const struct op_list *,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d);

struct op_list {
        struct op {
                op_t *op;
                size_t arg;
                size_t arg1;
                size_t arg2;
        } ops[16];
};

#define NEXT() do {                                                     \
                const struct op *pair =                            \
                        (const void *)((uintptr_t)ops + ip);            \
                                                                        \
                return pair->op(state, ops, ip + sizeof(struct op), \
                                i, pair->arg,                           \
                                a, b, c, d);                            \
        } while (0)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * Conditionally tail calls into the next loop iteration.
 */
static NO_INLINE void
iter(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{

        ip = 0;
        i += sizeof(__m256i);
        if (__builtin_expect(i >= arg, 0))
                return;

        NEXT();
}

#pragma GCC diagnostic ignored "-Wunused-function"

#define GEN_LDST(reg)                                                   \
        static NO_INLINE void                                           \
        load_##reg(struct filter_state *restrict state,                 \
                   const struct op_list *restrict ops,                  \
                   size_t ip, size_t i, size_t arg,                     \
                   __m256i a, __m256i b, __m256i c, __m256i d)          \
        {                                                               \
                                                                        \
                reg = *(__m256i *)((uintptr_t)state->ptrs[arg] + i);    \
                NEXT();                                                 \
        }                                                               \
                                                                        \
        static NO_INLINE void                                           \
        store_##reg(struct filter_state *restrict state,                \
                    const struct op_list *restrict ops,                 \
                    size_t ip, size_t i, size_t arg,                    \
                    __m256i a, __m256i b, __m256i c, __m256i d)         \
        {                                                               \
                                                                        \
                *(__m256i *)((uintptr_t)state->ptrs[arg] + i) = reg;    \
                NEXT();                                                 \
        }


GEN_LDST(a);
GEN_LDST(b);
GEN_LDST(c);
GEN_LDST(d);

#undef GEN_LDST

#define GEN_BIN(dst, src)                                               \
        static NO_INLINE void                                           \
        or_##dst##_##src(struct filter_state *restrict state,           \
                         const struct op_list *restrict ops,            \
                         size_t ip, size_t i, size_t arg,               \
                         __m256i a, __m256i b, __m256i c, __m256i d)    \
        {                                                               \
                                                                        \
                dst |= src;                                             \
                NEXT();                                                 \
        }                                                               \
                                                                        \
        static NO_INLINE void                                           \
        and_##dst##_##src(struct filter_state *restrict state,          \
                          const struct op_list *restrict ops,           \
                          size_t ip, size_t i, size_t arg,              \
                          __m256i a, __m256i b, __m256i c, __m256i d)   \
        {                                                               \
                                                                        \
                dst &= src;                                             \
                NEXT();                                                 \
        }                                                               \
                                                                        \
        static NO_INLINE void                                           \
        xor_##dst##_##src(struct filter_state *restrict state,          \
                          const struct op_list *restrict ops,           \
                          size_t ip, size_t i, size_t arg,              \
                          __m256i a, __m256i b, __m256i c, __m256i d)   \
        {                                                               \
                                                                        \
                dst ^= src;                                             \
                NEXT();                                                 \
        }

GEN_BIN(a, b);
GEN_BIN(a, c);
GEN_BIN(a, d);
GEN_BIN(b, a);
GEN_BIN(b, c);
GEN_BIN(b, d);
GEN_BIN(c, a);
GEN_BIN(c, b);
GEN_BIN(c, d);
GEN_BIN(d, a);
GEN_BIN(d, b);
GEN_BIN(d, c);
#undef GEN_BIN

#pragma GCC diagnostic pop

void
threaded_inreg(struct filter_state *restrict state)
{
        const struct op_list op_list = {
                .ops = {
                        {
                                .op = load_a,
                                .arg = 1,
                        },
                        {
                                .op = load_b,
                                .arg = 2,
                        },
                        {
                                .op = or_a_b,
                        },
                        {
                                .op = load_b,
                                .arg = 3,
                        },
                        {
                                .op = xor_a_b
                        },
                        {
                                .op = load_b,
                                .arg = 4,
                        },
                        {
                                .op = load_c,
                                .arg = 5,
                        },
                        {
                                .op = xor_b_c,
                        },
                        {
                                .op = and_a_b,
                        },
                        {
                                .op = store_a,
                                .arg = 0,
                        },
                        {
                                .op = iter,
                                .arg = 32 * state->count,
                        },
                },
        };

        if (state->count > 0) {
                const struct op_list *ops = &op_list;
                __m256i zero = { 0 };

                ops->ops[0].op(state, ops, sizeof(struct op), 0,
                               ops->ops[0].arg,
                               zero, zero, zero, zero);
        }

        return;
}

/**
 * Fused operators:
 *
 * a = neg_x[i] ^ (x0[i] | x1[i])
 * a &= neg_y[i] ^ y0[i]
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static NO_INLINE void
xor_or(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        const struct op *self =
                (const void *)((uintptr_t)ops + ip - sizeof(struct op));
        __m256i neg_x = *(__m256i *)((uintptr_t)state->ptrs[arg] + i);
        __m256i x0 = *(__m256i *)((uintptr_t)state->ptrs[self->arg1] + i);
        __m256i x1 = *(__m256i *)((uintptr_t)state->ptrs[self->arg2] + i);

        a = neg_x ^ (x0 | x1);
        NEXT();
}

static NO_INLINE void
acc_and_xor(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        const struct op *self =
                (const void *)((uintptr_t)ops + ip - sizeof(struct op));
        __m256i neg_y = *(__m256i *)((uintptr_t)state->ptrs[arg] + i);
        __m256i y0 = *(__m256i *)((uintptr_t)state->ptrs[self->arg1] + i);

        a &= neg_y ^ y0;
        NEXT();
}

static NO_INLINE void
store_iter(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        const struct op *self =
                (const void *)((uintptr_t)ops + ip - sizeof(struct op));

        *(__m256i *)((uintptr_t)state->ptrs[arg] + i) = a;

        ip = 0;
        i += sizeof(__m256i);
        if (__builtin_expect(i >= self->arg1, 0))
                return;

        NEXT();
}

#pragma GCC diagnostic pop

void
threaded_inreg_fused(struct filter_state *restrict state)
{
        const struct op_list op_list = {
                .ops = {
                        {
                                .op = xor_or,
                                .arg = 3,
                                .arg1 = 1,
                                .arg2 = 2,
                        },
                        {
                                .op = acc_and_xor,
                                .arg = 5,
                                .arg1 = 4,
                        },
                        {
                                .op = store_iter,
                                .arg = 0,
                                .arg1 = sizeof(__m256i) * state->count,
                        },
                },
        };

        if (state->count > 0) {
                const struct op_list *ops = &op_list;
                __m256i zero = { 0 };

                ops->ops[0].op(state, ops, sizeof(struct op), 0, 
                               ops->ops[0].arg,
                               zero, zero, zero, zero);
        }

        return;
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

static op_t wired_xor_or, wired_acc_and_xor, wired_store_iter;

#define WIRED_NEXT(next) do {                                           \
                const struct op *pair =                                 \
                        (const void *)((uintptr_t)ops + ip);            \
                                                                        \
                return next(state, ops, ip + sizeof(struct op),         \
                            i, pair->arg,                               \
                            a, b, c, d);                                \
        } while (0)


static NO_INLINE void
wired_xor_or(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        __m256i neg_x = *(__m256i *)((uintptr_t)state->ptrs[3] + i);
        __m256i x0 = *(__m256i *)((uintptr_t)state->ptrs[1] + i);
        __m256i x1 = *(__m256i *)((uintptr_t)state->ptrs[2] + i);

        a = neg_x ^ (x0 | x1);
        WIRED_NEXT(wired_acc_and_xor);
}

static NO_INLINE void
wired_acc_and_xor(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        __m256i neg_y = *(__m256i *)((uintptr_t)state->ptrs[5] + i);
        __m256i y0 = *(__m256i *)((uintptr_t)state->ptrs[4] + i);

        a &= neg_y ^ y0;
        WIRED_NEXT(wired_store_iter);
}

static NO_INLINE void
wired_store_iter(struct filter_state *restrict state, const struct op_list *restrict ops,
     size_t ip, size_t i, size_t arg,
    __m256i a, __m256i b, __m256i c, __m256i d)
{
        const struct op *self =
                (const void *)((uintptr_t)ops + ip - sizeof(struct op));

        *(__m256i *)((uintptr_t)state->ptrs[0] + i) = a;

        ip = 0;
        i += sizeof(__m256i);
        if (__builtin_expect(i >= self->arg1, 0))
                return;

        WIRED_NEXT(wired_xor_or);
}

#pragma GCC diagnostic pop

void
wired_inreg_fused(struct filter_state *restrict state)
{
        const struct op_list op_list = {
                .ops = {
                        {
                                .op = wired_xor_or,
                                .arg = 3,
                                .arg1 = 1,
                                .arg2 = 2,
                        },
                        {
                                .op = wired_acc_and_xor,
                                .arg = 5,
                                .arg1 = 4,
                        },
                        {
                                .op = wired_store_iter,
                                .arg = 0,
                                .arg1 = sizeof(__m256i) * state->count,
                        },
                },
        };

        if (state->count > 0) {
                const struct op_list *ops = &op_list;
                __m256i zero = { 0 };

                ops->ops[0].op(state, ops, sizeof(struct op), 0, 
                               ops->ops[0].arg,
                               zero, zero, zero, zero);
        }

        return;
}
