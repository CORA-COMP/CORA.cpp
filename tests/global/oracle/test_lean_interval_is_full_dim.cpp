// test_lean_interval_is_full_dim - Interval::is_full_dim against the CORALean oracle (its
// `isFullDim`)
//
// The flat box is checked in both modes where the oracle compares strictly by the machine's order at
// nearest. The oracles before batch 5 add a margin there that rounds away and read a zero width as
// positive, so with them the exact mode compares boxes of positive width only; in the sound mode a
// `true` of the oracle is a proof of full dimension and may not be contradicted.
//
// ORACLE: contSet/@interval/isFullDim.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.isFullDim", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            test::oracle::check(I.is_full_dim(), lean::Interval(I).is_full_dim());
            if (test::oracle::mode() == test::oracle::Mode::exact && !test::oracle::nearest_is_strict())
                continue;
            // a flat box: the first coordinate has no width
            const Tensor flat = I.inf().slice(-2, 0, 1), rest = I.inf().slice(-2, 1, n);
            const Interval F(Tensor::concat({flat, rest}, -2),
                             Tensor::concat({flat, I.sup().slice(-2, 1, n)}, -2));
            test::oracle::check(F.is_full_dim(), lean::Interval(F).is_full_dim());
        }
    });
    return test::finish("lean interval is_full_dim");
}
