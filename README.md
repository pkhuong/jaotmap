Experiments with hybrid interpretation methods for bitmap indexing
==================================================================

Test the correctness and performance of variously realistic
implementation methods for one sample query:

    (and (xor neg_x (or x0 x1))
         (xor neg_y y0))

It's a representative subquery of the sort of things I had to optimise
back in my ad tech days.  To scale this up, we'd add more bitmaps to
inner `or` reduction trees (e.g., `(or z0 z1 z2)`), and more `xor/or`
subexpressions in the outer `and` reduction tree.

The query is a bit small, but that's the reality of toy queries.  It
also doesn't have any repeated term, which avoids what tends to be a
weak point of interpretative methods.  On the other hard, that's also
true of most queries I've worked with, especially once minimised.

Methods:

1. `baseline` is the natural for loop with a fully fused body.  That's
   pretty much what I expect from a baseline JIT compiler built on top
   of a production code generator like LLVM.

2. `blocking` makes C calls to out-of-line functions for boolean
   operations on short blocks.

3. `fused_blocking` makes C calls to out-of-line functions, for larger
   fused operations, `(xor neg (or x y))` and `(and acc (xor neg x))`.

4. `specialised_widget` makes a C call to an out-of-line function that
   implements the body, with YMM arguments and return value.

5. `fully_specialised_widget` makes a C call to an out-of-line function
   that implements the body, including loads and stores.

6. `threaded_inreg` executes the loop with a register-based direct
   threaded VM, where each primitive is a base boolean operator;
   there is no blocking to amortise dispatch overhead, everything
   operates on `YMM` registers at a time.

7. `fused_threaded_inreg` adds superinstructions for the same fused
   operations as `fused_blocking`.

8. `wired_inreg_fused` hardcodes the continuation and some constants
   in `fused_threaded_inreg`.

The `fused_blocking` implementation is probably how I'd tend to write
a dynamic bitmap expression evaluator.  The benchmarked code does
benefit from hardcoding the dispatch with C calls, but otherwise shows
what overhead we can expect from strip mining and tiling our loops to
combine fused operators that work on `< L1D`-sized temporaries.
The "threaded" implementations are meant to explore dispatch overhead.

The `widget` implementations are unrealistically specialised, and
serve as clear lower bounds on the performance of YMM-at-a-time
dispatch.

The `threaded` implementations show what how well a realistic dispatch
mechanism can do, compared to the `widget`s.  I expect `threaded_inreg`
to do badly: it's pretty clear that we need bigger superinstructions
to amortise `NEXT`.  That's what `fused_threaded_inreg` implements,
and is close what I'd expect from a respectable implementation of a
YMM-at-a-time target.  Finally, `wired_inreg_fused` should be
representative of a simple runtime code generator, with complexity
similar to a runtime linker: copy some machine code, patch up
references, and write integer constants at a couple offsets.