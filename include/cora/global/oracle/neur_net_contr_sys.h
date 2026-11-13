// neur_net_contr_sys - lean::NeurNetContrSys, the closed loop of a nonlinear plant by CORALean
//
// Syntax:     lean::NeurNetContrSys loop(plant, layers, 1, dt);
//             auto sets = loop.reach(X0, U, t_final, alg, "singh");
// Operations: reach (neurNetContrSys.reach): the state sets at the ends of the control periods
// Outputs:    the oracle's float definition at a dtype of the unguaranteed ops (nearest)
// Per period: the network's image appended to the state set (CORA's aux_appendU2X), the steps of
//             alg lin with its error loop on the plant extended by the network's output, each end
//             set reduced, and the projection to the plant's states
// See also:   lean/nonlinear_sys.h, lean/neural_network.h, cont_dynamics/neur_net_contr_sys
//
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/oracle/neural_network.h"
#include "cora/global/oracle/request.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class NeurNetContrSys {
  public:
    /// The plant, its first `controls` inputs driven by the network of `layers`, sampled every dt.
    NeurNetContrSys(const cora::NonlinearSys &plant, std::vector<NnLayer> layers, int64_t controls,
                    double dt);

    /// The state sets at the ends of the periods 1 ... t_final / dt from X0 and the external input
    /// set U (none: no input), the plant by alg (the error loop) and the network at poly_method;
    /// none if the oracle refuses the run.
    std::optional<std::vector<Zonotope>> reach(const Zonotope &X0, const std::optional<Zonotope> &U,
                                               double t_final, const alg::Linearization &alg,
                                               const std::string &poly_method = "singh") const;

  private:
    int64_t n_, m_, external_;
    double dt_;
    Json f_;
    std::vector<NnLayer> layers_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
