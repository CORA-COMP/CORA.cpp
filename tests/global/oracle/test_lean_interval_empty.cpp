// test_lean_interval_empty - Interval::empty against the CORALean oracle, as agreement on emptiness
//
// The oracle's empty box is the crossed one (inf = 1 > sup = 0 in every coordinate), as CORALean
// has no n x 0 matrix; CORA.cpp's has no columns. The bounds cannot agree bit for bit, so the test
// checks that both denote the empty set and have the dimension n.

#include "cora/cont_set/interval/interval.h"
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
            const Interval mine = Interval::empty(n);
            test::oracle::check(mine.is_empty() && mine.dim() == n,
                                crossed && theirs.first.size(-2) == n);
        }
    });
    return test::finish("lean interval empty");
}
