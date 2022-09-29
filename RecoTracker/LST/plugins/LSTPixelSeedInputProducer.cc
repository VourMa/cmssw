#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/transform.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"

#include "Validation/RecoTrack/interface/trackFromSeedFitFailed.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "RecoTracker/LST/interface/LSTPixelSeedInput.h"

class LSTPixelSeedInputProducer : public edm::global::EDProducer<> {
public:
  explicit LSTPixelSeedInputProducer(edm::ParameterSet const& iConfig);
  ~LSTPixelSeedInputProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  size_t addStripMatchedHit(const SiStripMatchedRecHit2D& hit, std::vector<std::pair<int, int>>& monoStereoClusterList) const;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> mfToken_;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> seedTokens_;
  const edm::EDPutTokenT<LSTPixelSeedInput> lstPixelSeedInputPutToken_;
};

LSTPixelSeedInputProducer::LSTPixelSeedInputProducer(edm::ParameterSet const& iConfig)
    : mfToken_(esConsumes()),
      beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpot"))),
      lstPixelSeedInputPutToken_(produces<LSTPixelSeedInput>()) {
  seedTokens_ = edm::vector_transform(iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("seedTracks"),
                                      [&](const edm::InputTag& tag) { return consumes<edm::View<reco::Track>>(tag); });

}

void LSTPixelSeedInputProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));

  desc.addUntracked<std::vector<edm::InputTag>>(
    "seedTracks",
    std::vector<edm::InputTag>{edm::InputTag("seedTracksinitialStepSeeds"),
                               edm::InputTag("seedTrackshighPtTripletStepSeeds")});

  descriptions.addWithDefaultLabel(desc);
}


size_t LSTPixelSeedInputProducer::addStripMatchedHit(const SiStripMatchedRecHit2D& hit,
                                                     std::vector<std::pair<int, int>>& monoStereoClusterList) const {
  monoStereoClusterList.emplace_back(hit.monoHit().cluster().key(), hit.stereoHit().cluster().key());
  return monoStereoClusterList.size() - 1;
}


void LSTPixelSeedInputProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Setup
  const auto& mf = iSetup.getData(mfToken_);

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotToken_, recoBeamSpotHandle);
  reco::BeamSpot const& bs = *recoBeamSpotHandle;

  // Vector definitions
  std::vector<std::pair<int, int>> monoStereoClusterList; // FIXME: Needed?

  LSTPixelSeedInput pixelSeedInput;
  std::vector<float> see_px;
  std::vector<float> see_py;
  std::vector<float> see_pz;
  std::vector<float> see_dxy;
  std::vector<float> see_dz;
  std::vector<float> see_ptErr;
  std::vector<float> see_etaErr;
  std::vector<float> see_stateTrajGlbX;
  std::vector<float> see_stateTrajGlbY;
  std::vector<float> see_stateTrajGlbZ;
  std::vector<float> see_stateTrajGlbPx;
  std::vector<float> see_stateTrajGlbPy;
  std::vector<float> see_stateTrajGlbPz;
  std::vector<int> see_q;
  std::vector<unsigned int> see_algo;
  std::vector<std::vector<int>> see_hitIdx;

  for (size_t iColl = 0; iColl < seedTokens_.size(); ++iColl) {
    // Get seed tokens
    const auto& seedToken = seedTokens_[iColl];
    edm::Handle<edm::View<reco::Track>> seedTracksHandle;
    iEvent.getByToken(seedToken, seedTracksHandle);
    const auto& seedTracks = *seedTracksHandle;

    if (seedTracks.empty())
      continue;

    // Get seed algo
    edm::EDConsumerBase::Labels labels;
    labelsForToken(seedToken, labels);

    TString label = labels.module;
    //format label to match algoName
    label.ReplaceAll("seedTracks", "");
    label.ReplaceAll("Seeds", "");
    label.ReplaceAll("muonSeeded", "muonSeededStep");
    //for HLT seeds // FIXME: Needed?
    label.ReplaceAll("FromPixelTracks", "");
    label.ReplaceAll("PFLowPixel", "");
    label.ReplaceAll("hltDoubletRecovery", "pixelPairStep");  //random choice
    int algo = reco::TrackBase::algoByName(label.Data());

    // Get seed track refs
    edm::RefToBaseVector<reco::Track> seedTrackRefs;
    for (edm::View<reco::Track>::size_type i = 0; i < seedTracks.size(); ++i) {
      seedTrackRefs.push_back(seedTracks.refAt(i));
    }

    edm::ProductID id = seedTracks[0].seedRef().id();

    LogTrace("TrackingNtuple") << "NEW SEED LABEL: " << label << " size: " << seedTracks.size() << " algo=" << algo
                               << " ProductID " << id;

    for (size_t iSeed = 0; iSeed < seedTrackRefs.size(); ++iSeed) {
      const auto& seedTrackRef = seedTrackRefs[iSeed];
      const auto& seedTrack = *seedTrackRef;
      const auto& seedRef = seedTrack.seedRef();
      const auto& seed = *seedRef;


      if (seedRef.id() != id)
        throw cms::Exception("LogicError")
            << "All tracks in 'TracksFromSeeds' collection should point to seeds in the same collection. Now the "
               "element 0 had ProductID "
            << id << " while the element " << seedTrackRef.key() << " had " << seedTrackRef.id()
            << ". The source collection is " << labels.module << ".";

      const bool seedFitOk = !trackFromSeedFitFailed(seedTrack);

      const TrackingRecHit* lastRecHit = &*(seed.recHits().end() - 1);
      TrajectoryStateOnSurface tsos =
          trajectoryStateTransform::transientState(seed.startingState(), lastRecHit->surface(), &mf);
      auto const& stateGlobal = tsos.globalParameters();

      std::vector<int> hitIdx;
      for (auto const& hit : seed.recHits()) { // FIXME: Is this whole block of code needed?
        int subid = hit.geographicalId().subdetId();
        if (subid == (int)PixelSubdetector::PixelBarrel || subid == (int)PixelSubdetector::PixelEndcap) {
          const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&hit);
          const auto& clusterRef = bhit->firstClusterRef();
          const auto clusterKey = clusterRef.cluster_pixel().key();
          hitIdx.push_back(clusterKey);
        } else if (subid == (int)StripSubdetector::TOB || subid == (int)StripSubdetector::TID ||
                   subid == (int)StripSubdetector::TIB || subid == (int)StripSubdetector::TEC) {
          if (trackerHitRTTI::isMatched(hit)) {
            const SiStripMatchedRecHit2D* matchedHit = dynamic_cast<const SiStripMatchedRecHit2D*>(&hit);
            int monoIdx = matchedHit->monoClusterRef().key();
            int stereoIdx = matchedHit->stereoClusterRef().key();

            std::vector<std::pair<int, int>>::iterator pos =
                find(monoStereoClusterList.begin(), monoStereoClusterList.end(), std::make_pair(monoIdx, stereoIdx));
            size_t gluedIndex = -1;
            if (pos != monoStereoClusterList.end()) {
              gluedIndex = std::distance(monoStereoClusterList.begin(), pos);
            } else {
              // We can encounter glued hits not in the input
              // SiStripMatchedRecHit2DCollection, e.g. via muon
              // outside-in seeds (or anything taking hits from
              // MeasurementTrackerEvent). So let's add them here.
              gluedIndex = addStripMatchedHit(*matchedHit,/* stripMasks,*/ monoStereoClusterList);
            }
            hitIdx.push_back(gluedIndex);
          } else {
            const BaseTrackerRecHit* bhit = dynamic_cast<const BaseTrackerRecHit*>(&hit);
            const auto& clusterRef = bhit->firstClusterRef();
            unsigned int clusterKey;
            if (clusterRef.isPhase2()) {
              clusterKey = clusterRef.cluster_phase2OT().key();
            } else {
              clusterKey = clusterRef.cluster_strip().key();
            }
            hitIdx.push_back(clusterKey);
          }
        } else {
          LogTrace("TrackingNtuple") << " not pixel and not Strip detector";
        }
      }

      // Fill output
      see_px.push_back(seedFitOk ? seedTrack.px() : 0);
      see_py.push_back(seedFitOk ? seedTrack.py() : 0);
      see_pz.push_back(seedFitOk ? seedTrack.pz() : 0);
      see_dxy.push_back(seedFitOk ? seedTrack.dxy(bs.position()) : 0);
      see_dz.push_back(seedFitOk ? seedTrack.dz(bs.position()) : 0);
      see_ptErr.push_back(seedFitOk ? seedTrack.ptError() : 0);
      see_etaErr.push_back(seedFitOk ? seedTrack.etaError() : 0);
      see_stateTrajGlbX.push_back(stateGlobal.position().x());
      see_stateTrajGlbY.push_back(stateGlobal.position().y());
      see_stateTrajGlbZ.push_back(stateGlobal.position().z());
      see_stateTrajGlbPx.push_back(stateGlobal.momentum().x());
      see_stateTrajGlbPy.push_back(stateGlobal.momentum().y());
      see_stateTrajGlbPz.push_back(stateGlobal.momentum().z());
      see_q.push_back(seedTrack.charge());
      see_algo.push_back(algo);
      see_hitIdx.push_back(hitIdx);
    }
  }

  pixelSeedInput.setLSTPixelSeedTraits(see_px, see_py, see_pz, see_dxy, see_dz, see_ptErr, see_etaErr, see_stateTrajGlbX, see_stateTrajGlbY, see_stateTrajGlbZ, see_stateTrajGlbPx, see_stateTrajGlbPy, see_stateTrajGlbPz, see_q, see_algo, see_hitIdx);
  iEvent.emplace(lstPixelSeedInputPutToken_, std::move(pixelSeedInput));
}

DEFINE_FWK_MODULE(LSTPixelSeedInputProducer);
