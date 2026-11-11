// priv_box_cut - the box of the cut of a set by a polytope guard
//
// The cut keeps the guard's rows on the factors (ConZonotope::cut); its box is the one of
// CORALean's `ConZonotope.box`: `fm` from Fourier-Motzkin passes (never tighter than CORA's
// exact ranges), `cert` from the dual solutions of CORA's linear programs, checked by weak
// duality in directed order (as tight as CORA's ranges, outer for any solution), `both` the
// coordinatewise cut of the two.
//
// Syntax:   Interval I = priv_box_cut(cut, box, fm_passes);
// Inputs:   cut - the set with its rows, box - the method (alg::CutBox)
//           fm_passes - the passes of the methods fm and both
// Outputs:  I - a box that holds the cut; empty if the cut has no point
// See also: priv_guard_intersect_nondet_guard, ConZonotope::interval_cert
//
// CORA.cpp only: CORALean's float definition replaces CORA's boxing of a cut by programs
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "hybrid_dynamics/location/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_box_cut(const ConZonotope::Cut &cut, alg::CutBox box, int fm_passes) {
    switch (box) {
    case alg::CutBox::fm:
        return cut.interval_tight(fm_passes);
    case alg::CutBox::cert:
        return cut.interval_cert();
    case alg::CutBox::both:
        return cut.interval_cert_tight(fm_passes);
    }
    CORA_FAIL(Error::wrong_value, "priv_box_cut: unknown box method");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
