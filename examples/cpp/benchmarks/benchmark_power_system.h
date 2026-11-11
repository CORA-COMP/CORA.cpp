// benchmark_power_system - the power systems of TSPS25 from examples/benchmarks/data/TSPS25.json
//
// The equations are those powerSystem2cora wrote for MATLAB CORA (<case>_dyn.m, <case>_con.m),
// kept as text by examples/python/benchmarks/power_system_data.py. parse_equation turns one into an
// Expr (MATLAB syntax: x(i), y(i), u(i), pi, sin, cos, + - * / ^, unary minus); model builds the
// NonlinDASys of a case; subsystem_index is CORA's indexForSubsystems (the selection matrices of
// a subsystem in the whole system).
//
// Syntax:     PowerSystemData data = load_power_system();   NonlinDASys sys = data.model("IEEE14");
// See also:   benchmark_nonlinear_das_reach_arch23_powersystem.cpp, benchmark_data.h
//
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "benchmark_data.h"
#include "benchmark_hybrid.h"
#include "cora/cora.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

// The equations -----------------------------------------------------------------------------------

/// A recursive-descent parser of the equations of a generated file, with MATLAB's precedence: ^
/// binds tighter than unary minus, which binds tighter than * and /.
class EquationParser {
  public:
    // the text and the symbolic variables its x(i), y(i) and u(i) stand for
    EquationParser(const std::string &text, const std::vector<Expr> &x, const std::vector<Expr> &y,
                   const std::vector<Expr> &u)
        : s_(text), x_(x), y_(y), u_(u) {}

    // the whole text is one sum, nothing may follow it
    Expr parse() {
        // the sum, then nothing but blanks
        const Expr e = sum_();
        skip_();
        if (i_ != s_.size()) fail_("trailing characters");
        return e;
    }

  private:
    // the text, the variables and the position of the parser in the text
    const std::string &s_;
    const std::vector<Expr> &x_, &y_, &u_;
    std::size_t i_ = 0;

    // a syntax error names the offset
    [[noreturn]] void fail_(const std::string &what) const {
        throw std::runtime_error("equation: " + what + " at offset " + std::to_string(i_) + " of " +
                                 s_);
    }
    void skip_() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool take_(char c) {
        skip_();
        if (i_ < s_.size() && s_[i_] == c) {
            ++i_;
            return true;
        }
        return false;
    }
    // sum := product (('+' | '-') product)*
    Expr sum_() {
        Expr e = product_();
        for (;;) {
            if (take_('+'))
                e = e + product_();
            else if (take_('-'))
                e = e - product_();
            else
                return e;
        }
    }
    // product := unary (('*' | '/') unary)*
    Expr product_() {
        Expr e = unary_();
        for (;;) {
            if (take_('*'))
                e = e * unary_();
            else if (take_('/'))
                e = e / unary_();
            else
                return e;
        }
    }
    // unary := ('-' | '+') unary | power
    Expr unary_() {
        if (take_('-')) return -unary_();
        if (take_('+')) return unary_();
        return power_();
    }
    // power := atom ('^' exponent)*, left associative; the exponent is a number
    Expr power_() {
        Expr e = atom_();
        while (take_('^')) {
            skip_();
            const bool negative = take_('-');
            const double p = number_();
            e = p == std::round(p) ? pow(e, static_cast<int>(negative ? -p : p))
                                   : pow(e, negative ? -p : p);
        }
        return e;
    }
    double number_() {
        skip_();
        char *end = nullptr;
        const double v = std::strtod(s_.c_str() + i_, &end);
        if (end == s_.c_str() + i_) fail_("a number is expected");
        i_ = static_cast<std::size_t>(end - s_.c_str());
        return v;
    }
    // atom := number | pi | name '(' argument ')' | '(' sum ')'
    Expr atom_() {
        skip_();
        if (i_ >= s_.size()) fail_("unexpected end");
        if (take_('(')) {
            const Expr e = sum_();
            if (!take_(')')) fail_("')' is expected");
            return e;
        }
        if (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.')
            return Expr(number_());
        std::size_t j = i_;
        while (j < s_.size() && std::isalpha(static_cast<unsigned char>(s_[j]))) ++j;
        const std::string name = s_.substr(i_, j - i_);
        i_ = j;
        if (name == "pi") return Expr(3.141592653589793);
        if (!take_('(')) fail_("'(' is expected after " + name);
        // x(i), y(i), u(i) index (1-based); sin and cos take an expression
        if (name == "x" || name == "y" || name == "u") {
            const auto k = static_cast<std::size_t>(number_());
            if (!take_(')')) fail_("')' is expected");
            const std::vector<Expr> &v = name == "x" ? x_ : name == "y" ? y_ : u_;
            if (k < 1 || k > v.size()) fail_("the index " + std::to_string(k) + " of " + name);
            return v[k - 1];
        }
        const Expr arg = sum_();
        if (!take_(')')) fail_("')' is expected");
        if (name == "sin") return sin(arg);
        if (name == "cos") return cos(arg);
        fail_("the function " + name);
    }
};

// The cases ---------------------------------------------------------------------------------------

/// The bus sets of a case (CORA's bus struct), 1-based as in MATLAB.
struct Buses {
    std::vector<int> generator, load, slack, input, fault;
};

/// One power system case: its equations, its sizes and its buses.
struct PowerSystemCase {
    std::string name;
    int64_t states, inputs, constraints;
    std::vector<std::string> dyn, con;
    Buses bus;
    std::vector<double> VM;
};

/// The cases of the data file.
class PowerSystemData {
  public:
    explicit PowerSystemData(const Json &json) {
        // the equations, the sizes, the buses and the voltages of each case
        for (const Json &m : json["models"].array) {
            PowerSystemCase c{m["name"].string,
                              m["states"].integer(),
                              m["inputs"].integer(),
                              m["constraints"].integer(),
                              {},
                              {},
                              {},
                              {}};
            // the equations as text
            for (const Json &e : m["dyn"].array) c.dyn.push_back(e.string);
            for (const Json &e : m["con"].array) c.con.push_back(e.string);
            const Json &b = m["bus"];
            for (const auto &[key, set] :
                 {std::pair<const char *, std::vector<int> *>{"generator", &c.bus.generator},
                  {"load", &c.bus.load},
                  {"slack", &c.bus.slack},
                  {"input", &c.bus.input},
                  {"fault", &c.bus.fault}})
                for (const Json &v : b[key].array) set->push_back(static_cast<int>(v.integer()));
            for (const Json &v : m["VM"].array) c.VM.push_back(v.number);
            cases_.push_back(std::move(c));
        }
    }

    /// The case of that name.
    const PowerSystemCase &at(const std::string &name) const {
        for (const PowerSystemCase &c : cases_)
            if (c.name == name) return c;
        throw std::runtime_error("TSPS25: no case " + name);
    }

    /// The differential-algebraic system of a case: f and g parsed from the equations.
    NonlinDASys model(const std::string &name) const {
        // f and g are parsed once, at the construction
        const PowerSystemCase &c = at(name);
        const auto equations = [](const std::vector<std::string> &rows) {
            return [&rows](const std::vector<Expr> &x, const std::vector<Expr> &y,
                           const std::vector<Expr> &u) {
                std::vector<Expr> out;
                for (const std::string &row : rows)
                    out.push_back(EquationParser(row, x, y, u).parse());
                return out;
            };
        };
        return NonlinDASys(equations(c.dyn), equations(c.con), c.states, c.inputs, c.constraints);
    }

  private:
    std::vector<PowerSystemCase> cases_;
};

/// The data file of TSPS25 (CORACPP_BENCHMARK_DATA names another folder).
inline PowerSystemData load_power_system() {
    const std::string folder = std::getenv("CORACPP_BENCHMARK_DATA")
                                   ? std::getenv("CORACPP_BENCHMARK_DATA")
                                   : CORACPP_BENCHMARK_DATA;
    return PowerSystemData(JsonParser(read_file(folder + "/TSPS25.json")).parse());
}

// The decomposition -------------------------------------------------------------------------------

/// The 1-based position of `bus` in the list, as MATLAB's find.
inline int find_bus(const std::vector<int> &all, int bus) {
    for (std::size_t i = 0; i < all.size(); ++i)
        if (all[i] == bus) return static_cast<int>(i) + 1;
    throw std::runtime_error("indexForSubsystems: bus " + std::to_string(bus) +
                             " is not in the system");
}

/// CORA's indexForSubsystems (statesPerBus = 3): the selection matrices of the subsystem `sub` of
/// the system `full`, with the entries written as the function writes them.
inline DaeSubsystem subsystem_index(const PowerSystemData &data, const PowerSystemCase &full,
                                    const std::string &sub_name) {
    const PowerSystemCase &sub = data.at(sub_name);
    const int nGen = static_cast<int>(full.bus.generator.size());
    const int nBus = nGen + static_cast<int>(full.bus.load.size());
    const int nInput = static_cast<int>(full.bus.input.size());
    std::vector<int> all = full.bus.generator;
    all.insert(all.end(), full.bus.load.begin(), full.bus.load.end());
    std::vector<int> sub_all = sub.bus.generator;
    sub_all.insert(sub_all.end(), sub.bus.load.begin(), sub.bus.load.end());
    const int sGen = static_cast<int>(sub.bus.generator.size());
    const int sBus = sGen + static_cast<int>(sub.bus.load.size());
    const int sInput = static_cast<int>(sub.bus.input.size());
    // the matrices as row-major arrays, set at (row, column) 1-based
    const auto zeros = [](int64_t r, int64_t c) {
        return std::vector<double>(std::size_t(r * c), 0.0);
    };
    std::vector<double> X = zeros(sub.states, full.states),
                        Y = zeros(sub.constraints, full.constraints),
                        Uy = zeros(sub.inputs, full.constraints), Uv = zeros(sub.inputs, nGen),
                        Uu = zeros(sub.inputs, full.inputs);
    const auto set = [](std::vector<double> &M, int64_t columns, int row, int col) {
        M[std::size_t((row - 1) * columns + (col - 1))] = 1.0;
    };
    // the generators: three states each, the generator's input
    for (int i = 1; i <= sGen; ++i) {
        const int global = find_bus(all, sub.bus.generator[std::size_t(i - 1)]);
        for (int k = 1; k <= 3; ++k)
            set(X, full.states, sGen * (k - 1) + i, nGen * (k - 1) + global);
        set(Uu, full.inputs, i, global);
    }
    // the buses: the voltage and the angle, and the load input
    for (int i = 1; i <= sBus; ++i) {
        const int global = find_bus(all, sub_all[std::size_t(i - 1)]);
        if (global != 1) {
            set(Y, full.constraints, i, global);
            set(Y, full.constraints, sub.bus.slack.empty() ? sBus + i : sBus + i - 1,
                nBus + global - 1);
        } else {
            set(Y, full.constraints, 1, 1); // the slack bus has no angle
        }
        set(Uu, full.inputs, sGen + 2 * sInput + i, nGen + 2 * nInput + global);
    }
    // the input buses: a generator bus gives its constant voltage, a load bus its voltage
    for (int i = 1; i <= sInput; ++i) {
        const int bus = sub.bus.input[std::size_t(i - 1)];
        const int global = find_bus(all, bus);
        const bool generator = std::find(full.bus.generator.begin(), full.bus.generator.end(),
                                         bus) != full.bus.generator.end();
        if (global != 1) {
            if (generator)
                set(Uv, nGen, sGen + i, global);
            else
                set(Uy, full.constraints, sGen + i, global);
            set(Uy, full.constraints, sGen + sInput + i, nBus + global - 1);
        } else {
            set(Uv, nGen, sGen + 1, 1); // CORA writes the first input here, whichever i is
        }
    }
    const auto tensor = [](const std::vector<double> &M, int64_t r, int64_t c) {
        return Tensor::from_data(M, {r, c});
    };
    return {data.model(sub_name),
            tensor(X, sub.states, full.states),
            tensor(Y, sub.constraints, full.constraints),
            tensor(Uv, sub.inputs, nGen),
            tensor(Uy, sub.inputs, full.constraints),
            tensor(Uu, sub.inputs, full.inputs)};
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
