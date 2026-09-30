// benchmarkData - loads and runs an ARCH-COMP AFF instance from benchmarks/data
//
// The instance files (<instance>.json and <instance>.bin) are written by
// benchmarks/matlab/exportBenchmarkData.m from the original CORA scripts; their format is
// described in benchmarks/README.md. runInstance verifies one instance and prints the line
// 'benchmark,instance,result,time' of the original scripts (details go to stderr).
//
// Syntax:     cora::bench::runInstance("ISSF01_ISS01");
// See also:   contDynamics/linearSys/linearSys.h, specification/specification.h

#pragma once

#include "contDynamics/linearSys/linearSys.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::bench {

/// A JSON value; only what the instance files use (strings without escapes other than \" and \\).
struct Json {
    enum class Kind { Null, Number, String, Array, Object } kind = Kind::Null;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json &operator[](const std::string &key) const {
        const auto it = object.find(key);
        if (it == object.end()) throw std::runtime_error("benchmark json: missing key " + key);
        return it->second;
    }
    const Json &operator[](std::size_t i) const { return array.at(i); }
    bool isNull() const { return kind == Kind::Null; }
    int64_t integer() const { return static_cast<int64_t>(number); }
};

/// A recursive-descent parser for the JSON of the instance files.
class JsonParser {
  public:
    explicit JsonParser(const std::string &text) : s_(text) {}

    Json parse() {
        Json value = value_();
        skip_();
        if (i_ != s_.size()) fail_("trailing characters");
        return value;
    }

  private:
    const std::string &s_;
    std::size_t i_ = 0;

    [[noreturn]] void fail_(const std::string &what) const {
        throw std::runtime_error("benchmark json: " + what + " at offset " + std::to_string(i_));
    }
    // whitespace is skipped before every token
    void skip_() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    void expect_(char c) {
        skip_();
        if (i_ >= s_.size() || s_[i_] != c) fail_(std::string("expected '") + c + "'");
        ++i_;
    }
    std::string string_() {
        expect_('"');
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\') ++i_;
            out += s_.at(i_++);
        }
        expect_('"');
        return out;
    }
    Json value_() {
        skip_();
        Json v;
        if (i_ >= s_.size()) fail_("unexpected end");
        const char c = s_[i_];
        // objects and arrays take their members until the closing bracket
        if (c == '{') {
            v.kind = Json::Kind::Object;
            expect_('{');
            skip_();
            while (s_.at(i_) != '}') {
                const std::string key = string_();
                expect_(':');
                v.object[key] = value_();
                skip_();
                if (s_.at(i_) == ',') ++i_;
                skip_();
            }
            expect_('}');
        } else if (c == '[') {
            v.kind = Json::Kind::Array;
            expect_('[');
            skip_();
            while (s_.at(i_) != ']') {
                v.array.push_back(value_());
                skip_();
                if (s_.at(i_) == ',') ++i_;
                skip_();
            }
            expect_(']');
        } else if (c == '"') {
            // a string value
            v.kind = Json::Kind::String;
            v.string = string_();
        } else if (s_.compare(i_, 4, "null") == 0) {
            i_ += 4;
        } else {
            // anything else is a number, read with strtod for exact doubles
            char *end = nullptr;
            v.kind = Json::Kind::Number;
            v.number = std::strtod(s_.c_str() + i_, &end);
            if (end == s_.c_str() + i_) fail_("unexpected character");
            i_ = static_cast<std::size_t>(end - s_.c_str());
        }
        return v;
    }
};

/// An instance: the system, what verify is given, and what MATLAB CORA returned.
struct Instance {
    std::string benchmark, label;
    VerifyAlg alg;
    std::unique_ptr<LinearSys> sys;
    std::unique_ptr<VerifyParams> params;
    std::vector<Specification> specs;
    Json expected;
};

// Loading ---------------------------------------------------------------------------------------

inline std::string readFile(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("benchmark data: cannot read " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

/// The matrix of a "matrices" entry as dense row-major values.
inline std::vector<double> readMatrix(const Json &entry, const std::string &bin) {
    const int64_t rows = entry["shape"][0].integer(), cols = entry["shape"][1].integer();
    const int64_t nnz = entry["nnz"].integer(), offset = entry["offset"].integer();
    std::vector<double> dense(static_cast<std::size_t>(rows * cols), 0.0);
    const char *p = bin.data() + offset;
    if (entry["storage"].string == "dense") {
        std::copy_n(reinterpret_cast<const double *>(p), nnz, dense.begin());
        return dense;
    }
    // coordinates: int32 rows, int32 columns, float64 values (0-based)
    const auto *ri = reinterpret_cast<const int32_t *>(p);
    const auto *ci = ri + nnz;
    const auto *vals = reinterpret_cast<const double *>(p + 8 * nnz);
    for (int64_t k = 0; k < nnz; ++k) dense[ri[k] * cols + ci[k]] = vals[k];
    return dense;
}

/// A zonotope from its center (n, 1) and generators (n, m); no generators become one zero one.
inline Zonotope readZonotope(const Json &matrices, const std::string &bin, const std::string &c,
                             const std::string &G) {
    const int64_t n = matrices[c]["shape"][0].integer(), m = matrices[G]["shape"][1].integer();
    const Tensor center = Tensor::fromData(readMatrix(matrices[c], bin), {n, 1});
    if (m == 0) return Zonotope(center, Tensor::zeros({n, 1}));
    return Zonotope(center, Tensor::fromData(readMatrix(matrices[G], bin), {n, m}));
}

inline Tensor readTensor(const Json &matrices, const std::string &bin, const std::string &name) {
    const Json &entry = matrices[name];
    return Tensor::fromData(readMatrix(entry, bin),
                            {entry["shape"][0].integer(), entry["shape"][1].integer()});
}

// Specifications --------------------------------------------------------------------------------

/// One specification; a safe set may have several halfspaces, an unsafe set takes one.
inline Specification readSpec(const Json &spec) {
    std::vector<Halfspace> halfspaces;
    for (std::size_t k = 0; k < spec["A"].array.size(); ++k) {
        std::vector<double> row;
        for (const Json &x : spec["A"][k].array) row.push_back(x.number);
        const int64_t n = static_cast<int64_t>(row.size());
        halfspaces.push_back({Tensor::fromData(row, {n, 1}), spec["b"][k].number});
    }
    if (spec["type"].string == "safeSet") return Specification::safeSet(halfspaces);
    if (spec["type"].string != "unsafeSet")
        throw std::runtime_error("benchmark data: unknown specification type "
                                 + spec["type"].string);
    if (halfspaces.size() != 1)
        throw std::runtime_error("benchmark data: an unsafe set that is an intersection of "
                                 "halfspaces is not supported by Specification");
    return Specification::unsafeSet(halfspaces[0]);
}

/// Reads <name>.json and its binary file from benchmarks/data (or $CORACPP_BENCHMARK_DATA).
inline Instance loadInstance(const std::string &name) {
    const std::string folder = std::getenv("CORACPP_BENCHMARK_DATA")
                                   ? std::getenv("CORACPP_BENCHMARK_DATA")
                                   : CORACPP_BENCHMARK_DATA;
    const Json meta = JsonParser(readFile(folder + "/" + name + ".json")).parse();
    const std::string bin = readFile(folder + "/" + meta["data"].string);
    const Json &mat = meta["matrices"];

    // the algorithm name is the one of CORA's options.verifyAlg
    Instance inst;
    inst.benchmark = meta["benchmark"].string;
    inst.label = meta["label"].string;
    const std::string alg = meta["verifyAlg"].string;
    if (alg == "reachavoid:supportFunc") inst.alg = VerifyAlg::SupportFunc;
    else if (alg == "reachavoid:zonotope") inst.alg = VerifyAlg::Zonotope;
    else throw std::runtime_error("benchmark data: unknown verifyAlg " + alg);
    inst.sys = std::make_unique<LinearSys>(readTensor(mat, bin, "A"), readTensor(mat, bin, "B"),
                                           readTensor(mat, bin, "C"));
    inst.params = std::make_unique<VerifyParams>(VerifyParams{
        readZonotope(mat, bin, "R0c", "R0G"), readZonotope(mat, bin, "Uc", "UG"),
        meta["tFinal"].number});
    for (const Json &spec : meta["specs"].array) inst.specs.push_back(readSpec(spec));
    inst.expected = meta["expected"];
    return inst;
}

// Running ---------------------------------------------------------------------------------------

/// Verifies the instance, prints 'benchmark,instance,result,time' and, on stderr, the iterations,
/// steps and step size next to those of MATLAB. The time is tComp, else the wall clock.
inline void runInstance(const std::string &name) {
    const Instance inst = loadInstance(name);
    const auto start = std::chrono::steady_clock::now();
    const VerifyResult res = inst.sys->verify(*inst.params, inst.alg, inst.specs);
    const std::chrono::duration<double> wall = std::chrono::steady_clock::now() - start;

    std::cout << inst.benchmark << "," << inst.label << "," << res.verified << ","
              << (res.tComp > 0 ? res.tComp : wall.count()) << std::endl;
    std::cerr << "  iterations " << res.iterations << ", nrSteps " << res.nrSteps << ", timeStep "
              << res.timeStep;
    if (!inst.expected.isNull() && inst.expected.object.count("nrSteps"))
        std::cerr << "  (MATLAB: iterations " << inst.expected["iterations"].integer()
                  << ", nrSteps " << inst.expected["nrSteps"].integer() << ", timeStep "
                  << inst.expected["timeStep"].number << ")";
    std::cerr << std::endl;
}

/// The instances named on the command line, else all of the family.
inline std::vector<std::string> instanceNames(int argc, char **argv,
                                              const std::vector<std::string> &family) {
    if (argc < 2) return family;
    return std::vector<std::string>(argv + 1, argv + argc);
}

} // namespace cora::bench

// ---------------------------------------  END OF CODE  ---------------------------------------- //
