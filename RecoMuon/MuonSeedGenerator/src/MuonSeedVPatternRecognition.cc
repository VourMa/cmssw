#include "RecoMuon/MuonSeedGenerator/interface/MuonSeedVPatternRecognition.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

MuonSeedVPatternRecognition::MuonSeedVPatternRecognition(const edm::ParameterSet& pset)
    : enableDTMeasurement(pset.getParameter<bool>("EnableDTMeasurement")),
      enableCSCMeasurement(pset.getParameter<bool>("EnableCSCMeasurement")),
      enableME0Measurement(pset.getParameter<bool>("EnableME0Measurement")) {
  if (enableDTMeasurement)
    // the name of the DT rec hits collection
    theDTRecSegmentLabel = pset.getParameter<edm::InputTag>("DTRecSegmentLabel");

  if (enableCSCMeasurement)
    // the name of the CSC rec hits collection
    theCSCRecSegmentLabel = pset.getParameter<edm::InputTag>("CSCRecSegmentLabel");

  if (enableME0Measurement)
    theME0RecSegmentLabel = pset.getParameter<edm::InputTag>("ME0RecSegmentLabel");
}
