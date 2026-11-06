// test_zono_bundle_class - ZonoBundle: the constructor, its checks and the interface of a set
//
// CORA: contSet/@zonoBundle/zonoBundle.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/set.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two planar zonotopes: a box and a segment.
std::vector<Zonotope> two_members() {
    return {Zonotope(column({1.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})),
            Zonotope(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}))};
}

/// The bundle stores its members in order and is a set of the class with CORA's precedence.
void members_and_class(const std::string &b) {
    const ZonoBundle bundle(two_members());
    check(bundle.parallel_sets() == 2 && bundle.dim() == 2, b + ": two members of dimension 2");
    check(test::close(bundle.zonotopes()[1].generators(), {1.0, 1.0}, 0), b + ": members in order");
    const Set S(bundle);
    check(S.is<ZonoBundle>() && S.class_name() == "ZonoBundle", b + ": a Set holds it");
    check(precedence(ClassId::zono_bundle) == 100, b + ": CORA's precedence");
    check(S->is_bounded(), b + ": through the set interface");
}

/// The operations that are not ported yet say so.
void unported_operations(const std::string &b) {
    const ZonoBundle bundle(two_members());
    Rng rng(3);
    check(test::raises([&] { bundle.support_function(column({1.0, 0.0})); }, Error::not_supported),
          b + ": support_function");
    check(test::raises([&] { bundle.rand_point(2, rng); }, Error::not_supported),
          b + ": rand_point");
    check(test::raises([&] { bundle.project({0}); }, Error::not_supported), b + ": project");
    check(test::raises([&] { bundle.vertices(); }, Error::not_supported), b + ": vertices");
    check(test::raises([&] { bundle.vertices_3d(); }, Error::not_supported), b + ": vertices_3d");
}

/// A list that is empty, mixes dimensions or batch shapes is refused; a valid one passes with the
/// checks off.
void checks_and_their_switch() {
    const Zonotope Z1(column({0.0, 0.0}), Tensor({{1.0}, {1.0}})),
        Z3(column({0.0}), Tensor({{1.0}}));
    check(test::raises([] { ZonoBundle(std::vector<Zonotope>{}); },
                       Error::wrong_input_in_constructor),
          "ZonoBundle: an empty list");
    check(test::raises([&] { ZonoBundle({Z1, Z3}); }, Error::wrong_input_in_constructor),
          "ZonoBundle: members of two dimensions");
    check(test::raises([&] { ZonoBundle({Z1, Zonotope::stack({Z1, Z1})}); },
                       Error::wrong_input_in_constructor),
          "ZonoBundle: a batch and a single member");
    ChecksScope off(false);
    check(ZonoBundle({Z1, Z1}).parallel_sets() == 2, "ZonoBundle: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        members_and_class(b);
        unported_operations(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle class");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
