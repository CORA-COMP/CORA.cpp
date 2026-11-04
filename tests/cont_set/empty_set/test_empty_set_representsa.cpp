// test_empty_set_representsa - EmptySet representsa: which set types have an empty member
//
// CORA: contSet/@emptySet/representsa_.m
// PROPERTY: contSet/@emptySet/representsa_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

const char *const kTrue[] = {"capsule",    "conHyperplane", "conPolyZono", "conZonotope",
                             "ellipsoid",  "interval",      "polytope",    "polyZonotope",
                             "zonoBundle", "zonotope",      "convexSet",   "emptySet"};
const char *const kFalse[] = {"origin",     "point",         "halfspace", "probZonotope",
                              "hyperplane", "parallelotope", "fullspace"};

void the_table_of_cora(const std::string &b) {
    const EmptySet E(3);
    for (const char *type : kTrue) check(E.representsa(type), b + ": true for " + type);
    for (const char *type : kFalse) check(!E.representsa(type), b + ": false for " + type);
}

/// Random dimensions and tolerances (fixed seed): the answer depends on the type alone, and the
/// empty set is the complement of a whole space only when it represents "emptySet" and the whole
/// space does not.
void the_answer_depends_on_the_type_alone() {
    Rng rng(21);
    for (int trial = 0; trial < 20; ++trial) {
        const EmptySet E = EmptySet::generate_random(rng);
        const double tol = 1e-3 * trial;
        for (const char *type : kTrue)
            check(E.representsa(type, tol), std::string("true: ") + type);
        for (const char *type : kFalse)
            check(!E.representsa(type, tol), std::string("false: ") + type);
        check(E.representsa("emptySet") && !Fullspace(E.dim()).representsa("emptySet"),
              "the complement is not empty");
        check(E.representsa("fullspace") ==
                  Fullspace(E.dim()).complement().representsa("fullspace"),
              "the same answer for the complement of the whole space");
    }
}

/// An unknown type, a level set and a polygon are refused; with the checks off the answer is the
/// same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.representsa("circle"); }, Error::wrong_value), "an unknown type");
    check(test::raises([&] { E.representsa("levelSet"); }, Error::not_supported), "levelSet");
    check(test::raises([&] { E.representsa("polygon"); }, Error::not_supported), "polygon");
    check(test::raises([&] { E.representsa("zonotope", -1.0); }, Error::wrong_value), "tol < 0");
    ChecksScope off(false);
    check(E.representsa("zonotope") && !E.representsa("point"),
          "representsa: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_table_of_cora);
    the_answer_depends_on_the_type_alone();
    checks_and_their_switch();
    return test::finish("empty_set representsa");
}
