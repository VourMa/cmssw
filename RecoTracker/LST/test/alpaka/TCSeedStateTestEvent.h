#ifndef RecoTracker_LST_test_alpaka_TCSeedStateTestEvent_h
#define RecoTracker_LST_test_alpaka_TCSeedStateTestEvent_h

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/PixelSegmentParameters.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"
#include "RecoTracker/LSTCore/interface/TrackCandidatesHostCollection.h"

// Shared by the unit tests of the seed-state fit of the LST track candidates: a synthetic event in the formats the fit
// reads (the LST input with its OT rows, pLS hit slots and pixel seeds; the track candidates), built on the host from
// rows and candidates that refer to them, and the host references of what the fit must read from it. Plain types only.
namespace tcSeedStateTest {

  constexpr uint32_t kLayers = ::lst::Params_TC::kLayers;

  // a hit of the event: an OT row, or a hitsIT row (a pLS hit slot)
  struct Ref {
    bool it;
    uint32_t row;
  };
  constexpr Ref kEmpty{false, std::numeric_limits<uint32_t>::max()};
  inline bool isEmpty(Ref ref) { return !ref.it && ref.row == kEmpty.row; }
  inline bool operator==(Ref a, Ref b) { return a.it == b.it && a.row == b.row; }

  // an OT row (detId of the OT) or a hitsIT row (detId lst::kPixelModuleId for a pixel hit; for an OT hit of a pixel
  // track, the copy of its OT row with idx its row)
  struct Row {
    float x, y, z;
    float cov[6];
    uint32_t detId;
    uint32_t idx;
  };

  struct PixelTrack {
    uint32_t firstSlot;  // its first hitsIT row
    uint32_t nSlots, nHits, seedIdx;
  };

  struct Candidate {
    std::string label;
    ::lst::LSTObjType type;
    uint32_t pixelSeedIndex;
    Ref slots[kLayers][2];
  };

  struct Event {
    std::vector<Row> ot, it;
    std::vector<PixelTrack> pixelTracks;
    std::vector<Candidate> candidates;

    uint32_t nOT() const { return ot.size(); }
    uint32_t index(Ref ref) const { return isEmpty(ref) ? ::lst::kTCEmptyHitIdx : ref.it ? nOT() + ref.row : ref.row; }
    Row const& row(Ref ref) const { return ref.it ? it[ref.row] : ot[ref.row]; }
    Ref ref(uint32_t index) const { return index < nOT() ? Ref{false, index} : Ref{true, index - nOT()}; }

    Ref slot(uint32_t pixelTrack, uint32_t slot) const { return {true, pixelTracks[pixelTrack].firstSlot + slot}; }

    Candidate& addCandidate(std::string const& label, ::lst::LSTObjType type, uint32_t pixelSeedIndex) {
      Candidate candidate{label, type, pixelSeedIndex, {}};
      for (auto& slot : candidate.slots)
        slot[0] = slot[1] = kEmpty;
      candidates.push_back(candidate);
      return candidates.back();
    }

    // the pixel slots of a pT3 or a pT5 as the track-candidate builder fills them: (hit 0, hit 1), (hit 2, hit 3),
    // the third hit repeated for a pixel track of 3 slots (the pixel mini-doublets of
    // RecoTracker/LSTCore/src/alpaka/Segment.h, AddPixelMiniDoubletsToEventKernel); and its seed index
    void setPixelSlots(Candidate& candidate, uint32_t pixelTrack) const {
      const uint32_t nSlots = pixelTracks[pixelTrack].nSlots;
      candidate.slots[0][0] = slot(pixelTrack, 0);
      candidate.slots[0][1] = slot(pixelTrack, 1);
      candidate.slots[1][0] = slot(pixelTrack, 2);
      candidate.slots[1][1] = slot(pixelTrack, nSlots < 4 ? 2 : 3);
      candidate.pixelSeedIndex = pixelTracks[pixelTrack].seedIdx;
    }

    // the pixel slots of a pLS candidate, in the order of the track-candidate builder (TrackCandidate.h,
    // addpLSTrackCandidateToMemory): (hit 0, hit 2), (hit 1, hit 3)
    void setPLSSlots(Candidate& candidate, uint32_t pixelTrack) const {
      const uint32_t nSlots = pixelTracks[pixelTrack].nSlots;
      candidate.slots[0][0] = slot(pixelTrack, 0);
      candidate.slots[0][1] = slot(pixelTrack, 2);
      candidate.slots[1][0] = slot(pixelTrack, 1);
      candidate.slots[1][1] = slot(pixelTrack, nSlots < 4 ? 2 : 3);
      candidate.pixelSeedIndex = pixelTracks[pixelTrack].seedIdx;
    }
  };

  // The LST input of an event. The rows of the hits block behind the pLS slots hold what the fit must not read: a
  // position far outside the detector and a NaN covariance (the device LST input stores pLS kinematics there).
  inline ::lst::LSTInputHostCollection makeInput(Event const& event, float bField) {
    const uint32_t nOT = event.nOT(), nIT = event.it.size(), nPLS = event.pixelTracks.size();
    ::lst::LSTInputHostCollection input(cms::alpakatools::host(), nOT + nIT, nPLS, nIT);
    constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
    auto hits = input.view().hits();
    auto hitsIT = input.view().hitsIT();
    auto seeds = input.view().pixelSeeds();
    hits.nHitsOT() = nOT;
    hits.B() = bField;
    for (uint32_t h = 0; h < nOT; ++h) {
      Row const& row = event.ot[h];
      hits[h].xs() = row.x;
      hits[h].ys() = row.y;
      hits[h].zs() = row.z;
      hits[h].detid() = row.detId;
      hits[h].clustsize() = 2;
      for (int k = 0; k < 6; ++k)
        hits.cov()[h][k] = row.cov[k];
    }
    for (uint32_t k = 0; k < nIT; ++k) {
      Row const& row = event.it[k];
      hits[nOT + k].xs() = 1e4f + k;
      hits[nOT + k].ys() = -1e4f;
      hits[nOT + k].zs() = kNaN;
      hits[nOT + k].detid() = row.detId;
      hits[nOT + k].clustsize() = 1;
      for (int c = 0; c < 6; ++c)
        hits.cov()[nOT + k][c] = kNaN;
      hitsIT[k].idxs() = row.idx;
      hitsIT[k].xs() = row.x;
      hitsIT[k].ys() = row.y;
      hitsIT[k].zs() = row.z;
      for (int c = 0; c < 6; ++c)
        hitsIT.cov()[k][c] = row.cov[c];
    }
    for (uint32_t p = 0; p < nPLS; ++p) {
      seeds[p].firstHit() = nOT + event.pixelTracks[p].firstSlot;
      seeds[p].nHits() = event.pixelTracks[p].nHits;
      seeds[p].seedIdx() = event.pixelTracks[p].seedIdx;
    }
    return input;
  }

  // the track candidates of an event, candidate order[i] in row i
  inline ::lst::TrackCandidatesBaseHostCollection makeCandidates(Event const& event,
                                                                 std::vector<uint32_t> const& order) {
    const uint32_t n = order.size();
    ::lst::TrackCandidatesBaseHostCollection candidates(cms::alpakatools::host(), n);
    auto view = candidates.view();
    view.nTrackCandidates() = n;
    for (uint32_t i = 0; i < n; ++i) {
      Candidate const& candidate = event.candidates[order[i]];
      view.trackCandidateType()[i] = candidate.type;
      view.pixelSeedIndex()[i] = candidate.pixelSeedIndex;
      for (uint32_t slot = 0; slot < kLayers; ++slot)
        for (int k = 0; k < 2; ++k)
          view.hitIndices()[i][slot][k] = event.index(candidate.slots[slot][k]);
    }
    return candidates;
  }

  inline bool isPixel(Event const& event, uint32_t index) {
    return index >= event.nOT() && event.it[index - event.nOT()].detId == ::lst::kPixelModuleId;
  }

  // the LSTOTHits index of a hit, -1 for a pixel hit
  inline int64_t otIndex(Event const& event, uint32_t index) {
    if (index < event.nOT())
      return index;
    return isPixel(event, index) ? int64_t(-1) : int64_t(event.it[index - event.nOT()].idx);
  }

  // the encoded hit of a state on hit `index`: the LSTOTHits index of an OT hit; for a pixel hit, its position in its
  // pixel track from its slot in the pixel track that owns its hitsIT row
  inline uint32_t expectedStateHit(Event const& event, uint32_t index) {
    if (!isPixel(event, index))
      return ::lst::encodeStateHitOT(otIndex(event, index));
    const uint32_t row = index - event.nOT();
    for (auto const& track : event.pixelTracks)
      if (row >= track.firstSlot && row < track.firstSlot + track.nSlots)
        return ::lst::encodeStateHitPixel(
            ::lst::pixelSegmentHitOfSlot(row - track.firstSlot, track.nSlots, track.nHits));
    return ::lst::kInvalidU32Idx;
  }

  // the same float bits: a copied value, NaN included
  inline bool sameBits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

}  // namespace tcSeedStateTest

#endif
