// test_lean_interval_partition - Interval::partition against the CORALean oracle (its `partition`)
//
// The oracle answers one piece, given by the index of the cut in each coordinate; CORA.cpp
// returns all of them, the last coordinate counting fastest.
//
// ORACLE: contSet/@interval/partition.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.partition", [](test::oracle::Random &rnd) {
        for (const auto [n, splits] : {std::pair{1, 7}, {2, 3}, {3, 4}}) {
            const Interval I = rnd.box(n);
            const std::vector<Interval> pieces = I.partition(splits);
            for (std::size_t k = 0; k < pieces.size(); ++k) {
                std::vector<int64_t> ind(n);
                for (int64_t i = n - 1, rest = int64_t(k); i >= 0; --i, rest /= splits)
                    ind[i] = rest % splits;
                test::oracle::check(pieces[k], lean::Interval(I).partition(splits, ind));
            }
        }
    });
    return test::finish("lean interval partition");
}
