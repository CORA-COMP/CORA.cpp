// test_lean_empty_set - EmptySet against the CORALean oracle (its `emptySet.*` ops)
//
// The set holds no scalar, so every answer is exact (a dimension, a bool, a set that is given
// back). The other operand is an empty set, a box or a zonotope; the oracle reads its dimension,
// and its emptiness (a center without columns) for contains and isIntersecting. center, vertices
// and interval compare what the oracle has (no center, no vertex, CORALean's crossed empty box)
// with what CORA.cpp has (shape (n, 0)).
//
// ORACLE: contSet/@emptySet/and_.m
// ORACLE: contSet/@emptySet/center.m
// ORACLE: contSet/@emptySet/contains_.m
// ORACLE: contSet/@emptySet/convHull_.m
// ORACLE: contSet/@emptySet/empty.m
// ORACLE: contSet/@emptySet/emptySet.m
// ORACLE: contSet/@emptySet/interval.m
// ORACLE: contSet/@emptySet/isBounded.m
// ORACLE: contSet/@emptySet/isFullDim.m
// ORACLE: contSet/@emptySet/isIntersecting_.m
// ORACLE: contSet/@emptySet/isequal.m
// ORACLE: contSet/@emptySet/lift_.m
// ORACLE: contSet/@emptySet/not.m
// ORACLE: contSet/@emptySet/plus.m
// ORACLE: contSet/@emptySet/project.m
// ORACLE: contSet/@emptySet/projectHighDim_.m
// ORACLE: contSet/@emptySet/radius.m
// ORACLE: contSet/@emptySet/vertices_.m
// ORACLE: contSet/@emptySet/volume_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/empty_set.h"
#include "oracle_testing.h"

using namespace cora;
using test::oracle::check;
using test::oracle::Random;

namespace {

// The other operands of dimension n: the empty set, the empty box, a box and a zonotope.
std::vector<Set> aux_operands(Random &rnd, int64_t n) {
    return {EmptySet(n), Interval::empty(n), rnd.box(n), rnd.zonotope(n, 3)};
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    const auto each_dimension = [](auto body) {
        return [body](Random &rnd) {
            for (const int64_t n : {1, 2, 5, 20}) body(rnd, n);
        };
    };
    test::oracle::run("emptySet.empty", each_dimension([](Random &, int64_t n) {
                          check(EmptySet::empty(n).dim(), lean::EmptySet::empty(n).dim());
                          check(EmptySet(n).dim(),
                                lean::EmptySet::empty(n).dim()); // the constructor
                      }));
    test::oracle::run("emptySet.and", each_dimension([](Random &rnd, int64_t n) {
                          for (const Set &S : aux_operands(rnd, n))
                              check(EmptySet(n).intersect(*S).dim(),
                                    lean::EmptySet(n).intersect(*S).dim());
                      }));
    test::oracle::run("emptySet.plus", each_dimension([](Random &rnd, int64_t n) {
                          for (const Set &S : aux_operands(rnd, n))
                              check(EmptySet(n).plus(*S).dim(), lean::EmptySet(n).plus(*S).dim());
                      }));
    test::oracle::run("emptySet.convHull", each_dimension([](Random &rnd, int64_t n) {
                          const Interval I = rnd.box(n);
                          const Zonotope Z = rnd.zonotope(n, 3);
                          check(EmptySet(n).convex_hull(Set(I)).as<Interval>(),
                                lean::EmptySet(n).convex_hull(lean::Interval(I)));
                          // the zonotope comes back as it is, so it is compared exactly also in the
                          // sound mode (the oracle's certified containment of a zonotope is slow)
                          test::oracle::check_exact(
                              test::oracle::op_name(),
                              EmptySet(n).convex_hull(Set(Z)).as<Zonotope>(),
                              lean::EmptySet(n).convex_hull(lean::Zonotope(Z)));
                      }));
    test::oracle::run("emptySet.contains", each_dimension([](Random &rnd, int64_t n) {
                          for (const Set &S : aux_operands(rnd, n))
                              check(EmptySet(n).contains(*S), lean::EmptySet(n).contains(*S));
                      }));
    test::oracle::run("emptySet.isIntersecting", each_dimension([](Random &rnd, int64_t n) {
                          for (const Set &S : aux_operands(rnd, n))
                              check(EmptySet(n).is_intersecting(*S),
                                    lean::EmptySet(n).is_intersecting(*S));
                      }));
    test::oracle::run("emptySet.isequal", each_dimension([](Random &, int64_t n) {
                          // the oracle reads only the dimension of S, so S is an empty set, as in
                          // CORALean's row
                          for (const Set &S : {Set(EmptySet(n)), Set(Interval::empty(n))})
                              check(EmptySet(n).isequal(*S), lean::EmptySet(n).isequal(*S));
                      }));
    test::oracle::run("emptySet.isBounded", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).is_bounded(), lean::EmptySet(n).is_bounded());
                      }));
    test::oracle::run("emptySet.isFullDim", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).is_full_dim(), lean::EmptySet(n).is_full_dim());
                      }));
    test::oracle::run("emptySet.center", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).center().shape().back() == 0,
                                !lean::EmptySet(n).center().has_value());
                      }));
    test::oracle::run("emptySet.vertices", [](Random &) {
        const lean::Tensor V = lean::EmptySet(2).vertices();
        check(EmptySet(2).vertices().empty() && V.rows() == 2, V.cols() == 0);
    });
    // the oracle's empty box is the crossed one (inf > sup), CORA.cpp's has no columns: both
    // denote the empty set of dimension n
    test::oracle::run(
        "emptySet.interval", each_dimension([](Random &, int64_t n) {
            const std::pair<Tensor, Tensor> theirs = lean::EmptySet(n).interval().gather();
            bool crossed = true;
            for (int64_t i = 0; i < n; ++i)
                crossed = crossed && theirs.first.data()[i] > theirs.second.data()[i];
            const Interval mine = EmptySet(n).interval();
            check(mine.is_empty() && mine.dim() == n, crossed && theirs.first.size(-2) == n);
        }));
    test::oracle::run("emptySet.lift", each_dimension([](Random &, int64_t n) {
                          std::vector<int64_t> dims;
                          dims.reserve(n);
                          for (int64_t i = 0; i < n; ++i)
                              dims.push_back(n + 2 - 1 - i); // a permutation into n + 2
                          check(EmptySet(n).lift(n + 2, dims).dim(),
                                lean::EmptySet(n).lift(n + 2, dims).dim());
                      }));
    test::oracle::run("emptySet.project", each_dimension([](Random &, int64_t n) {
                          std::vector<int64_t> dims;
                          for (int64_t i = 0; i < n; i += 2) dims.push_back(i);
                          const Set P = EmptySet(n).project(dims);
                          check(P.as<EmptySet>().dim(), lean::EmptySet(n).project(dims).dim());
                      }));
    test::oracle::run("emptySet.projectHighDim", each_dimension([](Random &, int64_t n) {
                          std::vector<int64_t> dims;
                          dims.reserve(n);
                          for (int64_t i = 0; i < n; ++i) dims.push_back(n + 2 - 1 - i);
                          check(EmptySet(n).project_high_dim(n + 2, dims).dim(),
                                lean::EmptySet(n).project_high_dim(n + 2, dims).dim());
                      }));
    test::oracle::run("emptySet.radius", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).radius(), lean::EmptySet(n).radius());
                      }));
    test::oracle::run("emptySet.volume", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).volume(), lean::EmptySet(n).volume());
                      }));
    test::oracle::run("emptySet.not", each_dimension([](Random &, int64_t n) {
                          check(EmptySet(n).complement(), lean::EmptySet(n).complement());
                      }));
    return test::finish("lean empty_set");
}
