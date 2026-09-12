#ifndef RecoTracker_LSTCore_src_alpaka_MemoryProfiler_h
#define RecoTracker_LSTCore_src_alpaka_MemoryProfiler_h

#include <array>
#include <cstddef>
#include <string_view>
#include <type_traits>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/AllocatorConfig.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Per-event memory accounting for the LST collections, active only when the
  // memoryProfile flag is set. Tracks live bytes as a running quantity and keeps
  // a per-event high-water mark, separately for the device and host budgets.
  //
  // Peak is the headline number. The allocation sum is kept alongside it only so
  // the two can be compared: they coincide today because nothing is freed
  // mid-event, and they stop coinciding as soon as anything is.
  class MemoryProfiler {
  public:
    enum class Domain : unsigned { Device = 0, Host = 1 };
    static constexpr unsigned kNDomains = 2;

    // Whether the CMSSW caching allocator is actually in the allocation path.
    //
    // device_allocator_policy<DevCpu> reports Caching, but the CachedBufAlloc
    // specialisations for DevCpu with a CPU queue bypass the allocator entirely
    // and call alpaka::allocBuf / allocAsyncBuf directly
    // (HeterogeneousCore/AlpakaInterface/interface/CachedBufAlloc.h). So on the
    // CPU backends there is no caching and no bin rounding: reserved == requested,
    // and cacheStatus() stays at zero. Measured, not assumed.
    static constexpr bool kAllocatorInPath = not std::is_same_v<Device, alpaka::DevCpu>;

    struct Entry {
      std::string_view name;
      Domain domain;
      std::size_t slots = 0;      // element count, 0 when not meaningful
      std::size_t requested = 0;  // bytes asked for
      std::size_t reserved = 0;   // bytes the process actually holds
    };

    // Mirror of CachingAllocator::findBin, which is private and befriends only
    // AlpakaService (HeterogeneousCore/AlpakaInterface/interface/CachingAllocator.h).
    // Verified against cacheStatus() on the CUDA backend: 1000 -> 1024,
    // 16800000 -> 33554432. Unlike findBin this clamps instead of throwing, because
    // the harness must never change the outcome of a run.
    static constexpr std::size_t reservedFor(std::size_t bytes,
                                             cms::alpakatools::AllocatorConfig const& cfg = {}) {
      if constexpr (not kAllocatorInPath) {
        return bytes;
      } else {
        std::size_t minBinBytes = ipow(cfg.binGrowth, cfg.minBin);
        std::size_t maxBinBytes = ipow(cfg.binGrowth, cfg.maxBin);
        if (bytes < minBinBytes) {
          return minBinBytes;
        }
        if (bytes > maxBinBytes) {
          return bytes;  // findBin would throw; report the raw size instead
        }
        std::size_t binBytes = minBinBytes;
        while (binBytes < bytes) {
          binBytes *= cfg.binGrowth;
        }
        return binBytes;
      }
    }

    void recordAlloc(std::string_view name, Domain domain, std::size_t slots, std::size_t requested) {
      std::size_t reserved = reservedFor(requested);
      unsigned d = static_cast<unsigned>(domain);
      liveRequested_[d] += requested;
      liveReserved_[d] += reserved;
      sumRequested_[d] += requested;
      peakRequested_[d] = std::max(peakRequested_[d], liveRequested_[d]);
      peakReserved_[d] = std::max(peakReserved_[d], liveReserved_[d]);
      entries_.push_back(Entry{name, domain, slots, requested, reserved});
    }

    // All collections released together, as resetEventSync() does.
    void releaseAll() {
      liveRequested_.fill(0);
      liveReserved_.fill(0);
    }

    // Close the current event: the peaks stay readable until this is called.
    void endEvent() {
      releaseAll();
      peakRequested_.fill(0);
      peakReserved_.fill(0);
      sumRequested_.fill(0);
      entries_.clear();
    }

    std::size_t peakRequested(Domain d) const { return peakRequested_[static_cast<unsigned>(d)]; }
    std::size_t peakReserved(Domain d) const { return peakReserved_[static_cast<unsigned>(d)]; }
    std::size_t sumRequested(Domain d) const { return sumRequested_[static_cast<unsigned>(d)]; }
    std::size_t liveRequested(Domain d) const { return liveRequested_[static_cast<unsigned>(d)]; }
    std::vector<Entry> const& entries() const { return entries_; }

  private:
    static constexpr std::size_t ipow(unsigned base, unsigned exp) {
      std::size_t r = 1;
      for (unsigned i = 0; i < exp; ++i) {
        r *= base;
      }
      return r;
    }

    std::array<std::size_t, kNDomains> liveRequested_{};
    std::array<std::size_t, kNDomains> liveReserved_{};
    std::array<std::size_t, kNDomains> peakRequested_{};
    std::array<std::size_t, kNDomains> peakReserved_{};
    std::array<std::size_t, kNDomains> sumRequested_{};
    std::vector<Entry> entries_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
