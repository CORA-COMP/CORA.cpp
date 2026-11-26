// priv - the private functions of con_poly_zono
//
// Not for callers of the class: the block layouts of CORALean's ConPolyZono operations, shared by
// the operation files. Exponents are one column per generator, as in PolyZonotope.
//
// Syntax:   ConPolyZono S = priv_set(c, G, GI, E, cons);   Tensor R = priv_block_rows(A, B);
// See also: con_poly_zono.h
//
// CORA.cpp only: the block layouts of CORALean's ConPolyZono (genBlockDiag, the identifiers 1..p)
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/cont_set/con_poly_zono/con_poly_zono.h"

#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The identifiers 1, ..., p (CORA's default, and its choice after a block operation).
std::vector<int64_t> priv_ids(int64_t p);

/// The identity exponents of h generators on h factors.
Exponents priv_identity_exponents(int64_t h);

/// The set of the parts, stored as given, with the identifiers 1..p of cons's factors.
ConPolyZono priv_set(const Tensor &c, const Tensor &G, const Tensor &GI, Exponents E,
                     PolyLevelSet cons);

/// The generator rows of two sets on blocks of coordinates: A (h1, n1) over B (h2, n2) as
/// (h1 + h2, n1 + n2), +0 elsewhere (CORALean's genBlockDiag).
Tensor priv_block_rows(const Tensor &A, const Tensor &B);

/// The zonotope of a box as CORALean's Interval.zonotope: the center (inf + sup) / 2 and one
/// generator per side, of the radius max(sup - c, c - inf) (none dropped).
Zonotope priv_box_zonotope(const Interval &I);

/// x with every -0 made +0 (x + 0): the oracle's wire has no -0, so a zero that CORALean computes
/// as -0 is compared as +0.
Tensor priv_plus_zero(const Tensor &x);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
