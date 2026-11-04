// test_con_zonotope_isequal - ConZonotope::isequal: whether two sets are the same
//
// CORA: contSet/@conZonotope/isequal.m
//
// CORA authors:  Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the same data in another order of the factors; another representation of the
/// same set (the vertex test); a different set; empty sets; points; a zonotope and a box.
void isequal_known(const std::string &b) {
    const ConZonotope seg(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 2.0}}),
                          column({0.5}));
    const ConZonotope swapped(column({0.0, 0.0}), Tensor({{0.0, 1.0}, {1.0, 0.0}}),
                              Tensor({{2.0, 1.0}}), column({0.5}));
    check(seg.isequal(seg) && seg.isequal(swapped) && swapped.isequal(seg),
          b + ": equal up to the order of the factors");
    const ConZonotope other(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({{1.0, 2.0}}),
                            column({0.25}));
    check(!seg.isequal(other), b + ": another offset");
    // the unit square, with and without a pinned extra factor
    const ConZonotope square(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const ConZonotope pinned(column({0.0, 0.0}), Tensor({{1.0, 0.0, 5.0}, {0.0, 1.0, 5.0}}),
                             Tensor({{0.0, 0.0, 1.0}}), column({0.0}));
    check(square.isequal(pinned) && pinned.isequal(square), b + ": another representation");
    check(!square.isequal(ConZonotope(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 2.0}}))),
          b + ": another box");
    // empty sets are equal to each other only
    const ConZonotope none = ConZonotope::empty(2);
    check(none.isequal(ConZonotope::empty(2)) && !none.isequal(square) && !square.isequal(none),
          b + ": empty sets");
    const ConZonotope p(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}));
    check(p.isequal(ConZonotope(column({1.0, 2.0}), Tensor({{0.0, 0.0}, {0.0, 0.0}}))) &&
              !p.isequal(square),
          b + ": points");
    check(square.isequal(Zonotope(column({0.0, 0.0}), Tensor({{0.0, 1.0}, {1.0, 0.0}}))) &&
              square.isequal(Interval(column({-1.0, -1.0}), column({1.0, 1.0}))) &&
              !seg.isequal(Zonotope(seg.c(), seg.generators())),
          b + ": a zonotope and a box");
    check(!square.isequal(ConZonotope::origin(3)), b + ": another dimension");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(isequal_known);
    const ConZonotope a(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { a.isequal(a, -1.0); }, Error::wrong_value), "isequal: tol < 0");
    const ConZonotope batch(Tensor::zeros({2, 1, 1}), Tensor::zeros({2, 1, 1}));
    check(test::raises([&] { batch.isequal(batch); }, Error::not_supported), "isequal: a batch");
    ChecksScope off(false);
    check(a.isequal(a), "isequal: the same with the checks off");
    return test::finish("con_zonotope isequal");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
