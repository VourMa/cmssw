#ifndef RecoTracker_LSTCore_interface_TrackCandidatesDeviceCollection_h
#define RecoTracker_LSTCore_interface_TrackCandidatesDeviceCollection_h

#include "RecoTracker/LSTCore/interface/TrackCandidatesSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {
  using TrackCandidatesDeviceCollection = PortableCollection<TrackCandidatesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
#endif
