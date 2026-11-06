// test_polytope_set_difference - Polytope::set_difference: the pieces of P minus S
//
// CORA: contSet/@polytope/mldivide.m
// PROPERTY: contSet/@polytope/mldivide.m
//
// Property: a point is in P and not in S exactly if some piece holds it (the boundary aside)
//
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A square minus a smaller square: pieces that cover the frame.
void frames(const std::string &b) {
    const Polytope inner = Polytope(Interval(column({0.25, 0.25}), column({0.75, 0.75})));
    const std::vector<Polytope> pieces = test::square_h().set_difference(inner);
    check(pieces.size() == 4, b + ": four pieces around a square hole");
    auto in_pieces = [&](const Tensor &p) {
        for (const Polytope &piece : pieces)
            if (piece.contains(p, "exact", 1e-9)) return true;
        return false;
    };
    check(in_pieces(column({0.1, 0.5})) && in_pieces(column({0.9, 0.9})) &&
              !in_pieces(column({0.5, 0.5})),
          b + ": the frame, not the hole");
    // S outside of P: P itself, one piece; S around P: nothing
    const Polytope away = test::square_h().plus(column({5.0, 5.0}));
    check(test::square_h().set_difference(away).size() == 1, b + ": a disjoint S leaves P");
    check(test::square_h()
              .set_difference(Polytope(Interval(column({-1.0, -1.0}), column({2.0, 2.0}))))
              .empty(),
          b + ": S around P leaves nothing");
    check(Polytope::empty(2).set_difference(inner)[0].representsa("emptySet") &&
              test::square_h().set_difference(Polytope::empty(2)).size() == 1,
          b + ": the empty sets");
}

/// Random points are in the pieces exactly if they are in P and not in S.
void properties(const std::string &b) {
    Rng rng(4);
    bool same = true;
    const Polytope P1 = test::random_polytope(rng, 2, 4),
                   P2 = test::random_polytope(rng, 2, 3).plus(column({0.5, 0.5}));
    const std::vector<Polytope> pieces = P1.set_difference(P2);
    for (int k = 0; k < 60; ++k) {
        const Tensor x = column(test::random_direction(rng, 2));
        bool in_piece = false;
        for (const Polytope &piece : pieces)
            in_piece = in_piece || piece.contains(x, "exact", 1e-9);
        // the points that are strictly inside P1 and strictly outside P2 are covered
        const bool wanted = P1.contains(x, "exact", 0.0) && !P2.contains(x, "exact", 1e-9);
        same = same && (!wanted || in_piece) && (!in_piece || P1.contains(x, "exact", 1e-8));
    }
    check(same, b + ": the pieces cover P minus S and lie in P");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        frames(b);
        properties(b);
    });
    check(test::raises([] { test::square_h().set_difference(Polytope::inf(3)); },
                       Error::dimension_mismatch),
          "set_difference: dimensions");
    const Polytope plane(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}), Tensor({{1.0, 0.0}}),
                         column({0.5}));
    check(test::raises([&] { test::square_h().set_difference(plane); }, Error::not_supported),
          "set_difference: a lower-dimensional operand");
    ChecksScope off(false);
    check(test::square_h().set_difference(Polytope::empty(2)).size() == 1,
          "set_difference: the same with the checks off");
    return test::finish("polytope set_difference");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
