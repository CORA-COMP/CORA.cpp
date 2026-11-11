// test_taylm_interval - Taylm interval: the range with the monomial bounds
//
// CORA: contSet/@taylm/interval.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // x = a on [-1, 1]: a - a^2 is bounded termwise, [-1, 1] + [-1, 0] = [-2, 1]
        const std::vector<Taylm> a = Taylm::from_interval(Interval(column({-1.0}), column({1.0})));
        const Taylm f = a[0].minus(a[0].times(a[0]));
        check(close(f.bounds()[0], -2.0) && close(f.bounds()[1], 1.0),
              b + ": the monomial bounds do not cancel");
        const Interval I = f.interval();
        check(I.inf().shape() == std::vector<int64_t>({1, 1}) && close(I.sup(), {1.0}),
              b + ": a box of one dimension");
    });
    ChecksScope off(false);
    check(close(Taylm::constant(2.0, 1).interval().inf(), {2.0}), "interval: checks off");
    return test::finish("taylm interval");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
