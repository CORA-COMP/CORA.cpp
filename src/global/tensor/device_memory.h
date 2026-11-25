// device_memory - a device's free memory, its allocator's peak and its cache
//
// For the code that sizes its work by the memory a GPU has left (the verifier's root nets, CROWN's
// row chunks, the nodes of a batch) and that, out of memory, frees the cache before a smaller
// retry. On the CPU, and in a build without libtorch, there is nothing to ask: -1 and no-ops.
//
// Syntax:   const DeviceMemory m = device_memory(x.device_id());   free_device_cache(device);
// See also: cora/global/tensor/tensor.h, cora/global/backend/torch.h
//
// CORA.cpp only: CORA runs on MATLAB's memory
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <cstdint>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A device's memory in bytes: free (the driver's free memory and the allocator's unused cache),
/// in use by the allocator now and at its peak since the last reset; -1 where unknown (the CPU).
struct DeviceMemory {
    int64_t free = -1, allocated = -1, peak = -1;
};

/// The memory of `device` (libtorch's caching allocator on CUDA), -1 on the CPU.
DeviceMemory device_memory(Device device);

/// Returns the allocator's unused cache of `device` to the driver (torch.cuda.empty_cache).
void free_device_cache(Device device);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
