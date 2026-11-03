// test_lean_interval_lift - Interval::lift against the CORALean oracle (its `lift`)
//
// ORACLE: contSet/@interval/lift_.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;

int main() {
    if (!test::oracle::available()) return 0;
    test::oracle::run("interval.lift", [](test::oracle::Random &rnd) {
        for (const int n : {2, 5}) {
            const Interval I = rnd.box(n);
            std::vector<int64_t> dims(n);
            for (int j = 0; j < n; ++j) dims[j] = 2 * (n - 1 - j) + 1;
            test::oracle::check(I.lift(2 * n + 1, dims), lean::Interval(I).lift(2 * n + 1, dims));
        }
    });
    return test::finish("lean interval lift");
}
