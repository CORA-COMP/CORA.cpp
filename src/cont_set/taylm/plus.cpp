// plus - the sum and the difference of Taylor models, as CORA's taylm.plus and minus
//
// The polynomial parts and the remainders add; a number joins the constant term. The models must
// have the same variables. uminus negates both parts (the remainder swaps its bounds).
//
// Syntax:   T = A.plus(B);  T = A.minus(s);  T = A.uminus();
// Inputs:   B - a model of the variables of A, or a number s
// Outputs:  T - the model
// See also: times, interval
//
// CORA: contSet/@taylm/plus.m
// CORA: contSet/@taylm/minus.m
// CORA: contSet/@taylm/uminus.m
// CORA: contSet/@taylm/uplus.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Both models range over the same variables, else their sum has no meaning.
void aux_same_vars(const Taylm &a, const Taylm &b, const char *operation) {
    detail::Validation entry;
    CORA_CHECK(a.num_vars() == b.num_vars(), Error::dimension_mismatch, [&] {
        return std::string(operation) + ": the models have " + std::to_string(a.num_vars()) +
               " and " + std::to_string(b.num_vars()) + " variables";
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

Taylm Taylm::plus(const Taylm &S) const {
    detail::Entry entry;
    aux_same_vars(*this, S, "plus");
    Taylm out = *this;
    // the coefficients add monomial by monomial, the remainders as intervals
    for (const auto &[m, c] : S.terms_) out.terms_[m] += c;
    out.rem_lo_ += S.rem_lo_;
    out.rem_hi_ += S.rem_hi_;
    return out;
}

// A number joins the constant term ---------------------------------------------------------------

Taylm Taylm::plus(double s) const {
    detail::Entry entry;
    Taylm out = *this;
    out.terms_[Monomial(std::size_t(vars_), 0)] += s;
    return out;
}

Taylm Taylm::minus(const Taylm &S) const {
    detail::Entry entry;
    aux_same_vars(*this, S, "minus");
    return plus(S.uminus());
}

Taylm Taylm::minus(double s) const {
    detail::Entry entry;
    return plus(-s);
}

Taylm Taylm::uminus() const {
    detail::Entry entry;
    Taylm out = *this;
    for (auto &term : out.terms_) term.second = -term.second;
    // -[lo, hi] = [-hi, -lo]
    out.rem_lo_ = -rem_hi_;
    out.rem_hi_ = -rem_lo_;
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
