// test_zonotope_filter_out - Zonotope::filter_out: the zonotopes of a list that no other contains
//
// CORA: contSet/@zonotope/filterOut.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Tensor I2({{1.0, 0.0}, {0.0, 1.0}});
        const Zonotope big(column({0.0, 0.0}), I2 * 2.0), small(column({0.5, 0.0}), I2),
            away(column({10.0, 0.0}), I2);
        const std::vector<int64_t> kept = Zonotope::filter_out({small, big, away});
        check(kept == std::vector<int64_t>{1, 2}, b + ": the cell inside another is dropped");
        check(Zonotope::filter_out({big, away}) == std::vector<int64_t>{0, 1},
              b + ": disjoint cells stay");
        check(Zonotope::filter_out({big, big}).empty(),
              b + ": equal cells drop each other, as in CORA");
        check(Zonotope::filter_out({small}) == std::vector<int64_t>{0}, b + ": one cell stays");
    });
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    // wrong input is refused; the same call works with the checks off
    check(test::raises([] { Zonotope::filter_out({}); }, Error::wrong_value),
          "filter_out: no cell");
    check(
        test::raises(
            [&] { Zonotope::filter_out({Z, Zonotope(column({0.0, 0.0}), Tensor::zeros({2, 0}))}); },
            Error::wrong_value),
        "filter_out: cells of another dimension");
    ChecksScope off(false);
    check(Zonotope::filter_out({Z}).size() == 1, "filter_out: the same with the checks off");
    return test::finish("zonotope filter_out");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
