// test_lean_empty_set_interval - EmptySet::interval against the CORALean oracle (its `interval.empty`)
//
// CORALean's row for emptySet/interval is `Interval.empty`: the oracle's empty box is the crossed
// one (inf = 1 > sup = 0 in every coordinate), as CORALean has no n x 0 matrix; CORA.cpp's has no
// columns. The bounds cannot agree bit for bit, so the test checks that both denote the empty set
// and have the dimension n.
//
// ORACLE: contSet/@emptySet/interval.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.empty", [](test::oracle::Random &) {
        for (const int n : {1, 3, 20}) {
            const std::pair<Tensor, Tensor> theirs = lean::Interval::empty(n).gather();
            bool crossed = true;
            for (int i = 0; i < n; ++i)
                crossed = crossed && theirs.first.data()[i] > theirs.second.data()[i];
            const Interval mine = EmptySet(n).interval();
            test::oracle::check(mine.is_empty() && mine.dim() == n,
                                crossed && theirs.first.size(-2) == n);
        }
    });
    return test::finish("lean empty_set interval");
}
