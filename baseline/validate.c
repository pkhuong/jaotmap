#define RUN_ME /*
exec ${CC:-cc} ${CFLAGS:- -O3} -march=native -mtune=native -std=gnu11 -W -Wall      \
 noop.c baseline.c blocking.c fused_blocking.c specialised_widget.c threaded_inreg.c \
 $0 -o $(basename $0 .c)

*/
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "interface.h"

typedef void bv_fn_t(struct filter_state *);

struct vecs {
        __m256i *vecs[6];
};

static inline uint64_t
ticks_begin(void)
{
        uint32_t hi, lo;

        asm volatile("cpuid\n\t"
                     "rdtsc"
                     : "=d" (hi), "=a" (lo)
                     :: "%rbx", "%rcx");
        return ((uint64_t)hi << 32) + lo;
}

static inline uint64_t
ticks_end(void)
{
        uint32_t hi, lo;

        asm volatile("rdtscp\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "cpuid"
                     : "=r" (hi), "=r" (lo)
                     :: "%rax", "%rbx", "%rcx", "%rdx");
        return ((uint64_t)hi << 32) + lo;
}

static void
clear_caches(void)
{
        static const size_t bufsz = 32 * 1024 * 1024;
        static uint8_t *buf;

        if (buf == NULL) {
                buf = malloc(bufsz);
                memset(buf, 42, bufsz);
        }

        for (size_t i = 0; i < bufsz; i += 64)
                asm volatile("" :: "r"(buf[i]) : "memory");

        return;
}

static struct filter_state *
filter(bv_fn_t *fn, size_t count, struct vecs vecs)
{
        struct filter_state *ret;
        size_t vec_size = sizeof(__m256i) * count;
        int r;

        r = posix_memalign((void **)&ret, 32, sizeof(*ret));
        assert(r == 0);

        ret->count = count;
        for (size_t i = 0; i < 6; i++) {
                __m256i *copy;

                r = posix_memalign((void **)&copy, 32, vec_size);
                assert(r == 0);
                memcpy(copy, vecs.vecs[i], vec_size);
                ret->ptrs[i] = copy;
        }

        fn(ret);
        return ret;
}

static struct filter_state *
setup_empty_filter(size_t count)
{
        static size_t colour = 1;
        struct filter_state *ret;
        size_t vec_size = sizeof(__m256i) * count;
        int r;

        r = posix_memalign((void **)&ret, 32, sizeof(*ret));
        assert(r == 0);

        ret->count = count;
        for (size_t i = 0; i < 6; i++) {
                __m256i *copy;
                unsigned offset = 64 * colour++;

                r = posix_memalign((void **)&copy, 64, vec_size + offset);
                assert(r == 0);
                memset(copy, 42, vec_size);
                ret->ptrs[i] = copy;
        }

        return ret;
}

static void
destroy(struct filter_state *state)
{

        if (state == NULL)
                return;

        for (size_t i = 0; i < 6; i++)
                free(state->ptrs[i]);

        free(state);
        return;
}

static int
compare(bv_fn_t *control, bv_fn_t *test, size_t count, struct vecs vecs)
{
        struct filter_state *test_result = filter(test, count, vecs);
        struct filter_state *control_result = filter(control, count, vecs);
        size_t vec_size = sizeof(__m256i) * count;
        int r;

        r = memcmp(test_result->dst, control_result->dst, vec_size);
        destroy(test_result);
        destroy(control_result);
        return r;
}

static __m256i *
random_vec(size_t count)
{
        size_t vec_size = sizeof(__m256i) * count;
        __m256i *ret;
        int r;

        r = posix_memalign((void **)&ret, 32, vec_size);
        assert(r == 0);

        for (size_t i = 0; i < vec_size; i += sizeof(unsigned int)) {
                unsigned int bits = random();

                memcpy((char *)ret + i, &bits, sizeof(bits));
        }

        return ret;
}

static void
test_all(size_t count)
{
        struct vecs vecs;

        for (size_t i = 0; i < 6; i++)
                vecs.vecs[i] = random_vec(count);

        assert(compare(baseline, baseline, count, vecs) == 0);
        assert(compare(baseline, blocking, count, vecs) == 0);
        assert(compare(baseline, fused_blocking, count, vecs) == 0);
        assert(compare(baseline, specialised_widget, count, vecs) == 0);
        assert(compare(baseline, fully_specialised_widget, count, vecs) == 0);
        assert(compare(baseline, threaded_inreg, count, vecs) == 0);
        assert(compare(baseline, threaded_inreg_fused, count, vecs) == 0);
        assert(compare(baseline, wired_inreg_fused, count, vecs) == 0);
        
        for (size_t i = 0; i < 6; i++)
                free(vecs.vecs[i]);
        return;
}

static int
cmp_double(const void *vx, const void *vy)
{
        const double *x = vx;
        const double *y = vy;

        if (*x == *y)
                return 0;

        return (*x < *y) ? -1 : 1;
}

static double
time_fn(double offset, struct filter_state *state, bv_fn_t *fn, const char *name)
{
        enum { num_rep = 1000 };
        static double observations[num_rep];
        const size_t scale = sizeof(__m256i) * state->count;
        double ret;

        for (size_t i = 0; i < num_rep; i++) {
                uint64_t begin, end;

                clear_caches();

                begin = ticks_begin();
                fn(state);
                end = ticks_end();

                observations[i] = end - begin;
        }

        qsort(observations, num_rep, sizeof(observations[0]), cmp_double);

	/* Report the first percentile. */
        ret = (observations[num_rep / 100] / scale) - offset;
        if (name != NULL)
                printf("%32s\t%zu\t%.4f\n", name, scale, ret);
        return ret;
}

static void
time_all(size_t count)
{
        struct filter_state *state;
        double offset;

        state = setup_empty_filter(count);

        baseline(state);
        blocking(state);
        fused_blocking(state);
        specialised_widget(state);
        fully_specialised_widget(state);
        threaded_inreg(state);
        threaded_inreg_fused(state);

        fflush(NULL);
        fprintf(stderr, "==== n: %zu ====\n", count);
        offset = time_fn(0, state, noop, NULL);
        time_fn(offset, state, baseline, NULL);
        time_fn(offset, state, baseline, "baseline");
        time_fn(offset, state, blocking, "blocking");
        time_fn(offset, state, fused_blocking, "fused_blocking");
        time_fn(offset, state, specialised_widget, "specialised_widget");
        time_fn(offset, state, fully_specialised_widget, "fully_specialised_widget");
        time_fn(offset, state, threaded_inreg, "threaded_inreg");
        time_fn(offset, state, threaded_inreg_fused, "threaded_inreg_fused");
        time_fn(offset, state, wired_inreg_fused, "wired_inreg_fused");

        destroy(state);
        return;
}

int
main()
{

        clear_caches();

        test_all(32);
        test_all(64);
        test_all(128);
        test_all(1024 * 1024);

        time_all(32);
        time_all(128);
        time_all(1024);
        time_all(8 * 1024);
        time_all(16 * 1024);
        time_all(32 * 1024);
        time_all(64 * 1024);
        time_all(128 * 1024);
        time_all(256 * 1024);
        time_all(512 * 1024);
        time_all(1024 * 1024);
        time_all(8 * 1024 * 1024);
        return 0;
}
