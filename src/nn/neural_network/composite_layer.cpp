// composite_layer - CORA's composite layer: parallel branches whose outputs are added or joined
//
// A composite layer applies several branches (each a list of layers) to the same input and
// aggregates their outputs: 'add' sums them, 'concat' joins them along a dimension of their
// output shapes. This port's layers form a graph (a layer reads one source and any residuals), so
// the composite is lowered to the branches' layers and one aggregation layer that reads the last
// layer of each branch: for 'add' the identity of the first branch plus the others as identity
// residuals, for 'concat' each branch embedded at its place in the joined vector by a gather
// (zeros elsewhere). Points, batches of zonotopes and verify take the graph; a set evaluation of
// a chain does not (as for residuals in general). A branch without layers is the skip connection.
//
// Syntax:   NeuralNetwork nn(NeuralNetwork::composite_layer({b1, b2}, Aggregation::add, n));
// Inputs:   branches - blocks whose sources are -1 (the input) or their own layers
//           aggregation - add or concat
//           input_size - the number of values the branches read
//           output_shapes - the shape of each branch's output (default: its size)
//           concat_dim - the dimension joined
//           previous - the layer the composite reads
// Outputs:  block - the layers of all branches and the aggregation, with their activations
// See also: cora/nn/neural_network/neural_network.h, nonlinear_layers.cpp
// References: none (CORA's layer, no paper's algorithm)
//
// CORA: nn/layers/other/nnCompositeLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

#include <functional>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Layer = NeuralNetwork::Layer;

// The shapes --------------------------------------------------------------------------------------

/// The product of the shape's entries.
int64_t aux_count(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
}

/// The joined shape of the branches' shapes along dim (the ranks and the other dimensions match).
std::vector<int64_t> aux_joined(const std::vector<std::vector<int64_t>> &shapes, int64_t dim) {
    std::vector<int64_t> joined = shapes[0];
    for (std::size_t i = 1; i < shapes.size(); ++i)
        joined[std::size_t(dim)] += shapes[i][std::size_t(dim)];
    return joined;
}

/// The gather index of branch i: output position of each element of the branch's output (C
/// order) in the joined shape, -1 elsewhere.
std::vector<int64_t> aux_embedding(const std::vector<int64_t> &shape,
                                   const std::vector<int64_t> &joined, int64_t dim, int64_t shift) {
    std::vector<int64_t> index(std::size_t(aux_count(joined)), -1);
    const std::size_t rank = shape.size();
    for (int64_t j = 0; j < aux_count(shape); ++j) {
        // the multi-index of element j, then its place in the joined shape
        int64_t rest = j, out = 0, stride = 1;
        for (std::size_t d = rank; d-- > 0;) {
            const int64_t a = rest % shape[d];
            rest /= shape[d];
            out += (a + (int64_t(d) == dim ? shift : 0)) * stride;
            stride *= joined[d];
        }
        index[std::size_t(out)] = j;
    }
    return index;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The composite -----------------------------------------------------------------------------------

CORA_FROM_NUMBERS NeuralNetwork::Block NeuralNetwork::composite_layer(
    const std::vector<Block> &branches, Aggregation aggregation, int64_t input_size,
    const std::vector<std::vector<int64_t>> &output_shapes, int64_t concat_dim, int64_t previous) {
    detail::Entry entry;
    CORA_CHECK(!branches.empty() && input_size >= 1 &&
                   (output_shapes.empty() || output_shapes.size() == branches.size()),
               Error::wrong_input_in_constructor, [&] {
                   return "composite_layer: branches, an input size >= 1 and one output shape per "
                          "branch (or none), got " +
                          std::to_string(branches.size()) + " branches";
               });
    Block block;
    std::vector<int64_t> tails, sizes;
    for (const Block &branch : branches) {
        CORA_CHECK(branch.layers.size() == branch.activations.size(),
                   Error::wrong_input_in_constructor,
                   [] { return std::string("composite_layer: a layer and an activation each"); });
        // the layers of the branch from their local sources (-1: the input) to absolute ones
        const int64_t offset = previous + 1 + int64_t(block.layers.size());
        const auto absolute = [&](int64_t source) {
            return source < 0 ? previous : offset + source;
        };
        for (std::size_t j = 0; j < branch.layers.size(); ++j) {
            Layer layer = branch.layers[j];
            layer.source = layer.source == kPrevious ? (j == 0 ? previous : offset + int64_t(j) - 1)
                                                     : absolute(layer.source);
            for (Residual &r : layer.residuals) r.source = absolute(r.source);
            block.layers.push_back(layer);
            block.activations.push_back(branch.activations[j]);
        }
        // an empty branch is the input itself
        tails.push_back(branch.layers.empty() ? previous
                                              : offset + int64_t(branch.layers.size()) - 1);
        sizes.push_back(branch.layers.empty() ? input_size : branch.layers.back().outputs());
    }
    // the aggregation layer: the first branch is its source, the others its residuals
    Layer join;
    if (aggregation == Aggregation::add) {
        for (const int64_t size : sizes)
            CORA_CHECK(size == sizes[0], Error::wrong_input_in_constructor, [&] {
                return "composite_layer: the branches of an add have the same output size, got " +
                       std::to_string(sizes[0]) + " and " + std::to_string(size);
            });
        join = Layer::identity(sizes[0]);
        join.source = tails[0];
        for (std::size_t i = 1; i < tails.size(); ++i)
            join.residuals.push_back({tails[i], nullptr});
    } else {
        // the shapes: flat vectors if none are given
        std::vector<std::vector<int64_t>> shapes = output_shapes;
        if (shapes.empty())
            for (const int64_t size : sizes) shapes.push_back({size});
        for (std::size_t i = 0; i < shapes.size(); ++i)
            CORA_CHECK(aux_count(shapes[i]) == sizes[i], Error::wrong_input_in_constructor, [&] {
                return "composite_layer: the output shape " + shape_text(shapes[i]) +
                       " does not fit the " + std::to_string(sizes[i]) + " outputs of branch " +
                       std::to_string(i);
            });
        CORA_CHECK(concat_dim >= 0 && concat_dim < int64_t(shapes[0].size()),
                   Error::wrong_input_in_constructor, [&] {
                       return "composite_layer: the concat dimension is a dimension of the "
                              "output shapes, got " +
                              std::to_string(concat_dim);
                   });
        // the ranks agree, and so do the dimensions that are not joined
        for (const std::vector<int64_t> &shape : shapes) {
            CORA_CHECK(shape.size() == shapes[0].size(), Error::wrong_input_in_constructor, [&] {
                return "composite_layer: the branches' output shapes need the same rank, got " +
                       shape_text(shapes[0]) + " and " + shape_text(shape);
            });
            for (std::size_t d = 0; d < shape.size(); ++d)
                CORA_CHECK(int64_t(d) == concat_dim || shape[d] == shapes[0][d],
                           Error::wrong_input_in_constructor, [&] {
                               return "composite_layer: the output shapes differ in a dimension "
                                      "that is not joined: " +
                                      shape_text(shapes[0]) + " and " + shape_text(shape);
                           });
        }
        const std::vector<int64_t> joined = aux_joined(shapes, concat_dim);
        // each branch embedded at its place in the joined vector
        const Tensor zeros = Tensor::zeros({aux_count(joined)});
        int64_t shift = 0;
        for (std::size_t i = 0; i < shapes.size(); ++i) {
            const Layer embed =
                Layer::gather(aux_embedding(shapes[i], joined, concat_dim, shift), sizes[i], zeros);
            shift += shapes[i][std::size_t(concat_dim)];
            if (i == 0) {
                join = embed;
                join.source = tails[0];
            } else {
                join.residuals.push_back({tails[i], std::make_shared<Layer>(embed)});
            }
        }
    }
    // the aggregation is the last layer, with the identity as activation
    block.layers.push_back(join);
    block.activations.push_back(Activation::identity);
    return block;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
