// test_lean_polygon - the operations of Polygon that the CORALean oracle serves, bit for bit
//
// A polygon is CORALean's: a list of convex vertex pieces. Compared piece by piece and vertex by
// vertex in the stored order: plus (every pair, all vertex sums), mtimes, times, minus, or,
// linComb, project; the hull convHull and polytope; interval; splitIntoConvexSets; the bound of
// supportFunc; the Booleans contains, isIntersecting (a proof of disjointness), isequal, isConvex,
// isFullDim, isBounded and representsa (emptySet); empty. Not compared: vertices (CORA.cpp gives
// the outline of every piece, CORALean the stored vertices) and the clipping operations, which
// CORALean answers with boxes.
//
// ORACLE: contSet/@polygon/plus.m
// ORACLE: contSet/@polygon/mtimes.m
// ORACLE: contSet/@polygon/times.m
// ORACLE: contSet/@polygon/minus.m
// ORACLE: contSet/@polygon/or.m
// ORACLE: contSet/@polygon/linComb.m
// ORACLE: contSet/@polygon/project.m
// ORACLE: contSet/@polygon/convHull_.m
// ORACLE: contSet/@polygon/polytope.m
// ORACLE: contSet/@polygon/interval.m
// ORACLE: contSet/@polygon/splitIntoConvexSets.m
// ORACLE: contSet/@polygon/supportFunc_.m
// ORACLE: contSet/@polygon/contains_.m
// ORACLE: contSet/@polygon/isIntersecting_.m
// ORACLE: contSet/@polygon/isequal.m
// ORACLE: contSet/@polygon/isConvex.m
// ORACLE: contSet/@polygon/isFullDim.m
// ORACLE: contSet/@polygon/isBounded.m
// ORACLE: contSet/@polygon/representsa_.m
// ORACLE: contSet/@polygon/empty.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polygon/polygon.h"
#include "cora/global/oracle/polygon.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A piece of k random vertices about (cx, cy).
Polytope aux_piece(oracle::Random &rnd, double cx, double cy, int64_t k) {
    std::vector<double> x = rnd.matrix(k, 2).data();
    for (int64_t l = 0; l < k; ++l) x[2 * l] += cx, x[2 * l + 1] += cy;
    return Polytope(Tensor::from_data(x, {k, 2}));
}

/// A polygon of m random pieces about the diagonal, from `shift` on.
Polygon aux_polygon(oracle::Random &rnd, int m, double shift = 0) {
    std::vector<Polytope> pieces;
    pieces.reserve(std::size_t(m));
    for (int i = 0; i < m; ++i) pieces.push_back(aux_piece(rnd, shift + i, shift - i, 3 + i));
    return Polygon(pieces);
}

/// The pieces, one by one, and their number.
void aux_check(const std::string &op, const std::vector<Polytope> &mine,
               const std::vector<lean::Polytope> &theirs) {
    if (mine.size() != theirs.size()) {
        oracle::check(op + ": the number of pieces", int64_t(mine.size()), int64_t(theirs.size()));
        return;
    }
    for (std::size_t i = 0; i < mine.size(); ++i)
        oracle::check(op + "[" + std::to_string(i) + "]", mine[i], theirs[i]);
}

void aux_check(const std::string &op, const Set &mine, const lean::Polygon &theirs) {
    aux_check(op, mine.as<Polygon>().pieces(), theirs.pieces());
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the maps and sums, piece by piece
    oracle::run("polygon.plus", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2), B = aux_polygon(rnd, 2, 0.5);
        aux_check("plus", A.plus(B), lean::Polygon(A).plus(B));
    });
    oracle::run("polygon.mtimes", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        const Tensor M = rnd.matrix(2, 2);
        aux_check("mtimes", A.linear_map(M), lean::Polygon(A).linear_map(M));
    });
    oracle::run("polygon.times", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        const Tensor v = rnd.point(2);
        aux_check("times", A.times(v), lean::Polygon(A).times(v));
    });
    oracle::run("polygon.minus", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        const Tensor p = rnd.point(2);
        aux_check("minus", A.minus(p), lean::Polygon(A).minus(p));
        // plus a point is minus its negation
        aux_check("plus point", A.plus(p), lean::Polygon(A).minus(-p));
    });
    oracle::run("polygon.or", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2), B = aux_polygon(rnd, 1, 3);
        aux_check("or", A.unite(B), lean::Polygon(A).unite(B));
    });
    // the union and the pairwise hulls
    oracle::run("polygon.linComb", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2), B = aux_polygon(rnd, 2, 3);
        aux_check("linComb", A.linear_combination(B), lean::Polygon(A).linear_combination(B));
    });
    oracle::run("polygon.project", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        aux_check("project", A.project({1, 0}), lean::Polygon(A).project({1, 0}));
    });
    // the hull, the box, the pieces and the support bound
    oracle::run("polygon.convHull", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        oracle::check("convHull", A.convex_hull(), lean::Polygon(A).convex_hull());
    });
    oracle::run("polygon.polytope", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        oracle::check("polytope", A.polytope(), lean::Polygon(A).convex_hull());
    });
    oracle::run("polygon.interval", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        oracle::check("interval", A.interval(), lean::Polygon(A).interval());
    });
    oracle::run("polygon.splitIntoConvexSets", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        aux_check("splitIntoConvexSets", A.split_into_convex_sets(),
                  lean::Polygon(A).split_into_convex_sets());
    });
    // the support bound along several directions
    oracle::run("polygon.supportFunc", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        for (int k = 0; k < 4; ++k) {
            const Tensor d = rnd.direction(2);
            oracle::check(A.support_function(d), lean::Polygon(A).support_function(d));
        }
    });
    // the predicates
    oracle::run("polygon.contains", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 3);
        for (int k = 0; k < 8; ++k) {
            const Tensor p = rnd.point(2);
            oracle::check(A.contains(p), lean::Polygon(A).contains(p));
        }
    });
    oracle::run("polygon.isIntersecting", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        for (const double shift : {0.5, 2.0, 6.0}) {
            const Polygon B = aux_polygon(rnd, 2, shift);
            oracle::check(A.is_intersecting(B), !lean::Polygon(A).is_disjoint(B));
        }
    });
    // the pieces covering each other, and one piece holding all
    oracle::run("polygon.isequal", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2), B = aux_polygon(rnd, 2);
        oracle::check(A.isequal(A), lean::Polygon(A).isequal(A));
        oracle::check(A.isequal(B), lean::Polygon(A).isequal(B));
        const Polygon C = A.unite(Polygon({A.pieces()[0].linear_map(rnd.matrix(2, 2) * 0.1)}));
        oracle::check(A.isequal(C), lean::Polygon(A).isequal(C));
    });
    // one piece, and two that do not hold each other
    oracle::run("polygon.isConvex", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        oracle::check(A.is_convex(), lean::Polygon(A).is_convex());
        const Polygon one({A.pieces()[1]});
        oracle::check(one.is_convex(), lean::Polygon(one).is_convex());
    });
    oracle::run("polygon.isFullDim", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        oracle::check(A.is_full_dim(), lean::Polygon(A).is_full_dim());
        const Tensor t = rnd.point(2);
        const Polygon flat({Polytope(Tensor::stack({t, t * 2.0, t * 3.0}, 0))});
        oracle::check(flat.is_full_dim(), lean::Polygon(flat).is_full_dim());
    });
    // the constant, the empty test and the empty polygon
    oracle::run("polygon.isBounded", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        oracle::check(A.is_bounded(), lean::Polygon(A).is_bounded());
    });
    oracle::run("polygon.representsa", [](oracle::Random &rnd) {
        const Polygon A = aux_polygon(rnd, 2);
        oracle::check(A.representsa("emptySet"), lean::Polygon(A).represents_empty_set());
        oracle::check(Polygon::empty().representsa("emptySet"),
                      lean::Polygon(Polygon::empty()).represents_empty_set());
    });
    oracle::run("polygon.empty", [](oracle::Random &) {
        aux_check("empty", Polygon::empty().pieces(), lean::Polygon::empty().pieces());
    });
    return test::finish("lean polygon");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
