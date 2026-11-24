// test_specification_inverse - the inverse of safe and unsafe sets
//
// CORA: specification/@specification/inverse.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/polytope/polytope.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The box [lo1, hi1] x [lo2, hi2] as a polytope.
Polytope box(double lo1, double hi1, double lo2, double hi2) {
    return {Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
            vec({hi1, -lo1, hi2, -lo2})};
}

void inverses(const std::string &b) {
    const Polytope sq = box(0, 1, 0, 1), right = box(0.5, 2, 0, 1);
    // safe sets: the unsafe set of their intersection
    const std::vector<Specification> one =
        inverse({Specification::safe_set(sq), Specification::safe_set(right)});
    check(one.size() == 1 && one[0].type() == SpecType::unsafe_set,
          b + ": safe sets: one unsafe set");
    check(one[0].set().as<Polytope>().isequal(box(0.5, 1, 0, 1)), b + ": the intersection");
    // one unsafe set: the safe set
    const std::vector<Specification> safe = inverse({Specification::unsafe_set(sq)});
    check(safe.size() == 1 && safe[0].type() == SpecType::safe_set &&
              safe[0].set().as<Polytope>().isequal(sq),
          b + ": one unsafe set: the same set, safe");
    // several unsafe sets: unsafe sets covering the complement of their union
    const std::vector<Specification> many =
        inverse({Specification::unsafe_set(sq), Specification::unsafe_set(right)});
    // a point violates when one of the unsafe sets contains it
    const auto violated = [&](double x, double y) {
        bool bad = false;
        for (const Specification &spec : many) bad = bad || !spec.check_points(Tensor({{x, y}}));
        return bad;
    };
    // the points the sets allow and the ones they violate
    check(!violated(0.75, 0.5), b + ": inside both sets: allowed");
    // inside either set, nothing is violated
    check(!violated(0.25, 0.5), b + ": inside one set: allowed");
    check(!violated(1.5, 0.5), b + ": inside the other: allowed");
    check(violated(3.0, 3.0), b + ": outside both: violated");
    check(violated(-1.0, 0.5), b + ": left of both: violated");
    for (const Specification &spec : many)
        check(spec.type() == SpecType::unsafe_set, b + ": all unsafe");
}

void errors(const std::string &b) {
    const Polytope sq = box(0, 1, 0, 1);
    check(test::raises([&] { inverse({Specification::safe_set(sq).with_time(0.0, 1.0)}); },
                       Error::not_supported),
          b + ": timed");
    check(test::raises([&] { inverse({Specification::invariant(sq)}); }, Error::not_supported),
          b + ": an invariant");
    check(
        test::raises([&] { inverse({Specification::safe_set(sq), Specification::unsafe_set(sq)}); },
                     Error::not_supported),
        b + ": mixed");
    check(test::raises([&] { inverse({}); }, Error::not_supported), b + ": none");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        inverses(b);
        errors(b);
    });
    return test::finish("test_specification_inverse");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
