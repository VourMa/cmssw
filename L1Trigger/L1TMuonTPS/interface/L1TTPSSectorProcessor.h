#ifndef L1TTPSSECTORPROCESSOR_H
#define L1TTPSSECTORPROCESSOR_H

#include "DataFormats/L1TMuon/interface/L1MuCorrelatorHit.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TTPSSectorProcessor {
 public:
  typedef std::vector<edm::Ptr< l1t::TkMuon::L1TTTrackType > > TrackPtrVector;

  L1TTPSSectorProcessor(const edm::ParameterSet&);
  ~L1TTPSSectorProcessor();

  std::vector<l1t::TkMuon> process(const TrackPtrVector& ,const L1MuCorrelatorHitRefVector&);
  uint sector() const {
    return sectorNumber_;
  }


 private:
  typedef struct {
    int propagatedAngle;
    int propagatedBend; //only in software
    int propagatedEta;
    uint hasPhiB; //only in software
    uint propagatedSigmaAngle;
    uint propagatedSigmaBend;
    int etaIndex;   
    bool valid;
    uint propagatorIndex;
  } PropagationInfo;

  int deltaPhi(int phi1,int phi2);

  int stubPhi(const L1MuCorrelatorHitRef&);
  int trackPhi(const l1t::TkMuon&);
  int trackEta(const l1t::TkMuon&);
  int trackCurv(const l1t::TkMuon&);
  PropagationInfo propagate(const l1t::TkMuon&,uint);
  uint match(l1t::TkMuon&,const PropagationInfo&,const L1MuCorrelatorHitRefVector&,uint& pattern);
  bool processTrack(l1t::TkMuon&,const L1MuCorrelatorHitRefVector&);
  std::vector<l1t::TkMuon> clean(const std::vector<l1t::TkMuon>&);
  uint sectorNumber_;
  std::vector<uint> barrelSectors_;
  std::vector<uint> csc10DegreeChambers_;
  std::vector<uint> csc20DegreeChambers_;
  std::vector<uint> rpcEndcapChambers_;
  std::vector<uint> iRpcChambers_;
  int trackPhiLowerBound_;
  int trackPhiUpperBound_;
  int phiOffset_;

  int verbose_;
  double phiLSB_;
  double trackEtaLSB_;
  uint trackEtaShift_;
  double trackCurvLSB_;
  std::vector<uint> vetoIndex_;
  std::vector<uint> vetoPattern_;
  std::vector<uint> chiIndex_;
  std::vector<double> chiCut_;

  

};

#endif
