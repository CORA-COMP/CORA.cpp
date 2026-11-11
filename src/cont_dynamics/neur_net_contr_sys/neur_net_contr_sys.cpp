// neur_net_contr_sys - the class of neural network controlled systems: construction and its checks
//
// The closed loop is the plant extended by the held network outputs (CORA's
// aux_computeProperties for a nonlinearSys): f_ext(x, u_nn, u_ext) = (f(x, u_nn, u_ext), 0). The
// plant's inputs are ordered (u_nn, u_ext), so the variables of the plant's expressions are
// already those of the extended system, x_ext = (x, u_nn) and u_ext after them, and the extended
// system is built from the plant's expressions without substituting anything.
//
// See also: neur_net_contr_sys.h, reach.cpp
//
// CORA: contDynamics/@neurNetContrSys/neurNetContrSys.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The checks of the constructor, then the plant with the m controlled inputs as states of zero
/// derivative; the external inputs stay.
NonlinearSys aux_closed_loop(const NonlinearSys &plant, const NeuralNetwork &nn, double dt) {
    detail::Entry entry;
    // the sampling time, and the sizes of the network and the plant
    CORA_CHECK(dt > 0 && dt < 1e300, Error::wrong_input_in_constructor, [&] {
        return "NeurNetContrSys: the sampling time must be positive, got " + std::to_string(dt);
    });
    CORA_CHECK(nn.input_dim() == plant.dim(), Error::wrong_input_in_constructor, [&] {
        return "NeurNetContrSys: the network takes " + std::to_string(nn.input_dim()) +
               " inputs, the plant has " + std::to_string(plant.dim()) + " states";
    });
    CORA_CHECK(plant.input_dim() >= nn.output_dim(), Error::wrong_input_in_constructor, [&] {
        return "NeurNetContrSys: the network gives " + std::to_string(nn.output_dim()) +
               " outputs, the plant takes " + std::to_string(plant.input_dim()) + " inputs";
    });
    // the plant's expressions already index the extended system, so only the zeros are added
    const int64_t m = nn.output_dim();
    return {[&](const std::vector<Expr> &, const std::vector<Expr> &) {
                std::vector<Expr> f = plant.expressions();
                for (int64_t j = 0; j < m; ++j) f.emplace_back(0.0);
                return f;
            },
            plant.dim() + m, plant.input_dim() - m};
}

} // namespace


// ===========================================  MAIN  =========================================== //

NeurNetContrSys::NeurNetContrSys(NonlinearSys plant, NeuralNetwork nn, double dt)
    : plant_(std::move(plant)), nn_(std::move(nn)), dt_(dt),
      closed_loop_(aux_closed_loop(plant_, nn_, dt_)) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
