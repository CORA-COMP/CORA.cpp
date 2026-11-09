// linear_reset - the reset x+ = A x + B u + c of a transition, as CORA's linearReset
//
// The state before the jump x (n) and the input u (m) map to the state after it (n+, A is
// (n+, n)). Without B the input does not enter; without c there is no offset. evaluate maps a
// set (a zonotope or a zonotope bundle, with the input set) or points.
//
// Syntax:     LinearReset reset(A);   LinearReset reset(A, B, c);   reset.evaluate(X, U);
// Operations: evaluate (evaluate.cpp); the constructor checks its input (linear_reset.cpp)
// See also:   cora/hybrid_dynamics/transition/transition.h
//
// CORA: hybridDynamics/@linearReset/linearReset.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class LinearReset {
  public:
    /// x+ = A x + B u + c; A (n+, n), B (n+, m) and c (n+, 1) are optional single matrices.
    explicit LinearReset(Tensor A, std::optional<Tensor> B = std::nullopt,
                         std::optional<Tensor> c = std::nullopt);

    const Tensor &A() const noexcept { return A_; }
    const std::optional<Tensor> &B() const noexcept { return B_; }
    const std::optional<Tensor> &c() const noexcept { return c_; }

    /// The dimensions of the state before the jump, after it, and of the input (0 without B).
    int64_t pre_state_dim() const { return A_.shape().back(); }
    int64_t post_state_dim() const { return A_.shape()[0]; }
    int64_t input_dim() const { return B_ ? B_->shape().back() : 0; }

    /// The image A X + B U + c of a zonotope or a zonotope bundle X; U (a zonotope of the input)
    /// enters only with B.
    Set evaluate(const ContSet &X, const std::optional<Zonotope> &U = std::nullopt) const;

    /// The images A x + B u + c of the points x (n, N), with the inputs u (m, N) or (m, 1).
    Tensor evaluate(const Tensor &x, const std::optional<Tensor> &u = std::nullopt) const;

  private:
    Tensor A_;
    std::optional<Tensor> B_, c_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
