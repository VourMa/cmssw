#ifndef RecoMuon_TrackerSeedGenerator_SeparatingTSG_H
#define RecoMuon_TrackerSeedGenerator_SeparatingTSG_H

/** \class SeparatingTSG
 * Description:
 * composite TrackerSeedGenerator, which uses different TSG in different phase space of the track provided
 * concrete class must be implelemented (DualByEta ,...) to provide the TSG selection.
 *
 * \author Jean-Roch Vlimant
 */

#include "RecoMuon/TrackerSeedGenerator/interface/CompositeTSG.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class TrackerTopology;

class SeparatingTSG : public CompositeTSG {
public:
  SeparatingTSG(const edm::ParameterSet &pset, edm::ConsumesCollector &IC);
  ~SeparatingTSG() override;

  void trackerSeeds(const TrackCand &, const TrackingRegion &, const TrackerTopology *, BTSeedCollection &) override;

  virtual unsigned int selectTSG(const TrackCand &, const TrackingRegion &) = 0;

private:
  std::string theCategory;
};

#endif
