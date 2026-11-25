// device_memory - a device's free memory, its allocator's peak and its cache
//
// Syntax:   const DeviceMemory m = device_memory(x.device_id());   free_device_cache(device);
// See also: global/tensor/device_memory.h
//
// CORA.cpp only: CORA runs on MATLAB's memory
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "global/tensor/device_memory.h"

#ifdef CORACPP_TORCH
#include <ATen/DeviceAccelerator.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

#ifdef CORACPP_TORCH
/// Whether libtorch's accelerator is CUDA and `device` one of its devices.
bool aux_cuda(Device device) {
    if (device.type != Device::Type::cuda) return false;
    const auto accelerator = at::accelerator::getAccelerator(false);
    return accelerator.has_value() && *accelerator == c10::DeviceType::CUDA;
}
#endif

} // namespace


// ===========================================  MAIN  =========================================== //

// The memory --------------------------------------------------------------------------------------

DeviceMemory device_memory([[maybe_unused]] Device device) {
    DeviceMemory m;
#ifdef CORACPP_TORCH
    if (!aux_cuda(device)) return m;
    // the allocator's counts and the driver's free memory (the allocator's cache is free to it)
    const auto stats = at::accelerator::getDeviceStats(device.index);
    const auto [driver_free, total] = at::accelerator::getMemoryInfo(device.index);
    (void)total;
    const auto &reserved = stats.reserved_bytes[0], &allocated = stats.allocated_bytes[0];
    m.allocated = allocated.current;
    m.peak = allocated.peak;
    m.free = static_cast<int64_t>(driver_free) + (reserved.current - allocated.current);
#endif
    return m;
}

// The cache ---------------------------------------------------------------------------------------

void free_device_cache([[maybe_unused]] Device device) {
#ifdef CORACPP_TORCH
    if (aux_cuda(device)) at::accelerator::emptyCache();
#endif
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
