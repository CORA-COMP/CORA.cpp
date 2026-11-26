// con_poly_zono - the constructors and conversions of the constrained polynomial zonotope
//
// The set composes a PolyZonotope (c, G, E, GI, id) and a PolyLevelSet (A, b, EC), stored as given
// (CORA's constructor keeps its arguments too). The conversions follow CORALean: a zonotope keeps
// its generators as dependent ones on identity exponents, a constrained zonotope its rows A x = b
// on the identity monomials, a box goes through its zonotope (CORALean's Interval.zonotope, one
// generator per side). Also: origin, empty, zonotope, isequal, representsa, display.
//
// Syntax:   ConPolyZono cPZ(c, G, E, A, b, EC, GI, id);   ConPolyZono cPZ(Z);
// Inputs:   c - (n)
//           G - (h, n)
//           E, EC - exponents
//           A - (m, q)
//           b - (m)
//           GI - (r, n)
//           id - (p)
// Outputs:  cPZ - the constrained polynomial zonotope
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/conPolyZono.m
// CORA: contSet/@conPolyZono/origin.m
// CORA: contSet/@conPolyZono/empty.m
// CORA: contSet/@conPolyZono/isequal.m
// CORA: contSet/@conPolyZono/representsa_.m
// CORA: contSet/@conPolyZono/zonotope.m
// CORA: contSet/@conPolyZono/polyZonotope.m
// CORA: contSet/@conPolyZono/display.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"
#include "global/format.h"

#include <cmath>
#include <sstream>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether a and b have the same shape and agree within tol entry by entry.
bool aux_close(const Tensor &a, const Tensor &b, double tol) {
    if (a.shape() != b.shape()) return false;
    const std::vector<double> x = a.data(), y = b.data();
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!(std::abs(x[i] - y[i]) <= tol)) return false;
    return true;
}

/// A single set of the class `what` (its center (n)): a batch is refused.
void aux_single(const Tensor &center, const char *what) {
    detail::Entry entry;
    CORA_CHECK(center.ndim() == 1, Error::dimension_mismatch,
               [&] { return std::string(what) + ": a ConPolyZono is a single set, not a batch"; });
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constructors ----------------------------------------------------------------------------

ConPolyZono::ConPolyZono(Tensor c, Tensor G, Exponents E, Tensor A, Tensor b, Exponents EC,
                         Tensor GI, const std::vector<int64_t> &id)
    : ConPolyZono(PolyZonotope(std::move(c), std::move(G), std::move(GI), std::move(E), id,
                               detail::Verbatim{}),
                  PolyLevelSet(std::move(A), std::move(b), std::move(EC), int64_t(id.size()))) {}

ConPolyZono::ConPolyZono(PolyZonotope poly, PolyLevelSet cons)
    : ContSet(kClassId), poly_(std::move(poly)), cons_(std::move(cons)) {
    detail::Entry entry;
    CORA_CHECK(cons_.num_vars() == poly_.num_factors(), Error::wrong_input_in_constructor, [&] {
        return "ConPolyZono: the constraint is on " + std::to_string(cons_.num_vars()) +
               " factors, the set has " + std::to_string(poly_.num_factors());
    });
}

// a zonotope: its generators dependent, the identity exponents, no constraint
ConPolyZono::ConPolyZono(const Zonotope &Z)
    : ConPolyZono(priv_set(Z.center(), Z.generators(), Z.center().zeros_like({0, Z.dim()}),
                           priv_identity_exponents(Z.nr_generators()),
                           PolyLevelSet::none(Z.center(), Z.nr_generators()))) {
    aux_single(Z.center(), "Zonotope");
}

ConPolyZono::ConPolyZono(const PolyZonotope &poly)
    : ConPolyZono(poly, PolyLevelSet::none(poly.center(), poly.num_factors())) {}

// a constrained zonotope: the rows A x = b on the identity monomials (b + 0, as CORALean's be - 0)
ConPolyZono::ConPolyZono(const ConZonotope &cz)
    : ConPolyZono(priv_set(cz.c(), cz.generators(), cz.c().zeros_like({0, cz.dim()}),
                           priv_identity_exponents(cz.generators().shape()[0]),
                           PolyLevelSet(cz.A(), priv_plus_zero(cz.b()),
                                        priv_identity_exponents(cz.generators().shape()[0]),
                                        cz.generators().shape()[0]))) {
    aux_single(cz.c(), "ConZonotope");
}

ConPolyZono::ConPolyZono(const Interval &I) : ConPolyZono(priv_box_zonotope(I)) {}

CORA_FROM_NUMBERS ConPolyZono ConPolyZono::origin(int64_t n) {
    const Tensor c = Tensor::zeros({n});
    return priv_set(c, c.zeros_like({0, n}), c.zeros_like({0, n}), {}, PolyLevelSet::none(c, 0));
}

// the row 0 = 1 of no monomial
CORA_FROM_NUMBERS ConPolyZono ConPolyZono::empty(int64_t n) {
    const Tensor c = Tensor::zeros({n});
    return priv_set(c, c.zeros_like({0, n}), c.zeros_like({0, n}), {},
                    PolyLevelSet(c.zeros_like({1, 0}), c.ones_like({1}), {}, 0));
}

// The conversions and the queries -------------------------------------------------------------

Zonotope ConPolyZono::zonotope() const {
    return {c(), Tensor::concat({generators(), independent_generators()}, 0)};
}

bool ConPolyZono::isequal(const ConPolyZono &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c().dtype(), "isequal");
    // the structure, then every number, field by field
    if (exponents() != S.exponents() || constraint_exponents() != S.constraint_exponents())
        return false;
    return aux_close(c(), S.c(), tol) && aux_close(generators(), S.generators(), tol) &&
           aux_close(independent_generators(), S.independent_generators(), tol) &&
           aux_close(A(), S.A(), tol) && aux_close(b(), S.b(), tol);
}

bool ConPolyZono::representsa(const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c().dtype(), "representsa");
    // a point: no generator in either block
    if (type == "point")
        return generators().shape()[0] == 0 && independent_generators().shape()[0] == 0;
    if (type == "emptySet") {
        // a row without monomials and with b != 0 holds nowhere
        const std::vector<double> rhs = b().data();
        if (cons_.num_monomials() > 0) return false;
        for (const double v : rhs)
            if (std::abs(v) > tol) return true;
        return false;
    }
    CORA_FAIL(Error::not_supported, "representsa: the type \"" + type +
                                        "\" is not supported for a constrained polynomial "
                                        "zonotope (\"point\", \"emptySet\")");
}

std::string ConPolyZono::display_impl() const {
    // the counts of the parts, in one line
    std::ostringstream out;
    out << "constrained polynomial zonotope of dimension " << dim() << ": "
        << generators().shape()[0] << " dependent generators over " << ids().size() << " factors, "
        << independent_generators().shape()[0] << " independent generators, " << cons_.num_rows()
        << " constraints over " << cons_.num_monomials() << " monomials";
    return out.str();
}

Tensor ConPolyZono::rand_point_impl(int64_t, Rng &) const {
    CORA_FAIL(Error::not_supported,
              "rand_point: a point of a constrained polynomial zonotope needs a solution of the "
              "polynomial constraints (CORA's randPoint_ is not ported)");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
