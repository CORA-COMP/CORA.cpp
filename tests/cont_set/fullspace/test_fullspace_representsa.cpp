// test_fullspace_representsa - fullspace representsa: the whole space is an interval and convex
//
// CORA: contSet/@fullspace/representsa_.m
// PROPERTY: contSet/@fullspace/representsa_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

const char *const kTrue[] = {"interval", "convexSet", "fullspace"};
const char *const kFalse[] = {"origin",      "point",        "capsule",       "conHyperplane",
                              "conPolyZono", "conZonotope",  "ellipsoid",     "halfspace",
                              "polytope",    "polyZonotope", "probZonotope",  "zonoBundle",
                              "zonotope",    "hyperplane",   "parallelotope", "emptySet"};

void the_table_of_cora(const std::string &b) {
    const Fullspace F(3);
    for (const char *type : kTrue) check(F.representsa(type), b + ": true for " + type);
    for (const char *type : kFalse) check(!F.representsa(type), b + ": false for " + type);
}

/// In R^0 CORA's emptyObject rule answers: everything but origin, point and parallelotope.
void the_empty_space_is_every_type_but_a_point(const std::string &b) {
    const Fullspace F(0);
    for (const char *type : {"origin", "point", "parallelotope"})
        check(!F.representsa(type), b + ": R^0 is not " + type);
    for (const char *type : {"zonotope", "emptySet", "fullspace", "interval", "polytope"})
        check(F.representsa(type), b + ": R^0 is " + type);
    check(F.representsa("levelSet") && F.representsa("polygon"), b + ": no refusal in R^0");
}

/// Random dimensions and tolerances (fixed seed): the answer depends on the type alone and the
/// whole space and the empty set answer oppositely for the types that tell them apart.
void the_answer_depends_on_the_type_alone() {
    Rng rng(22);
    for (int trial = 0; trial < 20; ++trial) {
        Fullspace F = Fullspace::generate_random(rng);
        const double tol = 1e-3 * trial;
        for (const char *type : kTrue)
            check(F.representsa(type, tol), std::string("true: ") + type);
        for (const char *type : kFalse)
            check(!F.representsa(type, tol), std::string("false: ") + type);
        const EmptySet E = F.complement();
        for (const char *type : {"fullspace", "emptySet"})
            check(F.representsa(type) != E.representsa(type), std::string("opposite: ") + type);
        for (const char *type : {"origin", "point"})
            check(!F.representsa(type) && !E.representsa(type), std::string("neither: ") + type);
    }
}

/// An unknown type, a level set and a polygon are refused; with the checks off the answer is the
/// same.
void checks_and_their_switch() {
    const Fullspace F(2);
    check(test::raises([&] { F.representsa("circle"); }, Error::wrong_value), "an unknown type");
    check(test::raises([&] { F.representsa("levelSet"); }, Error::not_supported), "levelSet");
    check(test::raises([&] { F.representsa("polygon"); }, Error::not_supported), "polygon");
    check(test::raises([&] { F.representsa("interval", -1.0); }, Error::wrong_value), "tol < 0");
    ChecksScope off(false);
    check(F.representsa("interval") && !F.representsa("zonotope"),
          "representsa: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_table_of_cora);
    test::for_each_backend(the_empty_space_is_every_type_but_a_point);
    the_answer_depends_on_the_type_alone();
    checks_and_their_switch();
    return test::finish("fullspace representsa");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
