// neur_net_contr_sys - lean::NeurNetContrSys, the closed loop of a nonlinear plant by CORALean
//
// The plant goes as the tree of its expressions (its inputs u_0 ... the network's first, then the
// external ones), the network as its layers; CORALean joins, steps and projects.
//
// Syntax:     lean::NeurNetContrSys loop(plant, layers, 1, dt);   loop.reach(X0, U, t_final, alg);
// Inputs:     plant - the open-loop plant, layers - the network, controls - the inputs it drives
// Outputs:    the state sets at the ends of the control periods
// See also:   lean/nonlinear_sys.cpp, lean/neural_network.cpp
//
// CORA.cpp only: the oracle client of CORALean's neurNetContrSys.reach
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neur_net_contr_sys.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {


// ===========================================  MAIN  =========================================== //

NeurNetContrSys::NeurNetContrSys(const cora::NonlinearSys &plant, std::vector<NnLayer> layers,
                                 int64_t controls, double dt)
    : n_(plant.dim()), m_(controls), external_(plant.input_dim() - controls), dt_(dt),
      layers_(std::move(layers)) {
    f_ = Json::array();
    for (const Expr &e : plant.expressions())
        f_.push(Json::parse(e.to_json(int(n_), [](double x) { return Tensor::encode(x); })));
}

std::optional<std::vector<Zonotope>>
NeurNetContrSys::reach(const Zonotope &X0, const std::optional<Zonotope> &U, double t_final,
                       const alg::Linearization &alg, const std::string &poly_method) const {
    // the network in wire form, then the plant, the sets and the options of the run
    std::string dtype;
    const Json layers = layers_json(layers_, dtype);
    std::vector<Operand> request = {{"n", n_},
                                    {"m", m_},
                                    {"mExt", external_},
                                    {"f", Wire(f_, "")},
                                    {"p", 53},
                                    {"layers", Wire(layers, dtype)},
                                    {"poly_method", poly_method},
                                    {"X0", X0}};
    if (U) request.push_back({"U", *U});
    for (const Operand &o : std::vector<Operand>{{"dt", dt_},
                                                 {"tFinal", t_final},
                                                 {"timeStep", alg.time_step},
                                                 {"taylorTerms", alg.taylor_terms},
                                                 {"zonotopeOrder", alg.zonotope_order}})
        request.push_back(o);
    if (alg.linear_step == alg::LinearStep::closed_form)
        request.push_back({"linearStep", "closedForm"});
    // a refused run is an error of the oracle: no sets
    try {
        return ask("neurNetContrSys.reach", request).get<std::vector<Zonotope>>("timePoint");
    } catch (const std::runtime_error &) {
        return std::nullopt;
    }
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
