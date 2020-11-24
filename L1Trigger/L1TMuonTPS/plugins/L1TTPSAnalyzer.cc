#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <TTree.h>
#include <TString.h>


class L1TTPSAnalyzer : public edm::EDAnalyzer {
    public:
        explicit L1TTPSAnalyzer(const edm::ParameterSet&);
        ~L1TTPSAnalyzer();
      
    private:
        virtual void beginJob() ;
        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void endJob() ;

        uint64_t event_;
        int nMuonTPS_,nGen_,nTracks_;
        TTree *tree_;

        struct Particles {
            int n;
            std::vector<int> id;
            std::vector<float> pt, eta, phi, dr;
            void makeBranches(const std::string &prefix, TTree *tree) {
                tree->Branch(("n_"+prefix).c_str(),   &n, (prefix+"/I").c_str());
                tree->Branch((prefix+"_id").c_str(),  &id );
                tree->Branch((prefix+"_pt").c_str(),  &pt );
                tree->Branch((prefix+"_eta").c_str(), &eta);
                tree->Branch((prefix+"_phi").c_str(), &phi);
                tree->Branch((prefix+"_dr").c_str(),  &dr );
             }
            void clear() {
                id.clear(); pt.clear(); eta.clear(); phi.clear(); dr.clear();
            }
            void fill(const auto &c, float DR) {
                id.push_back(c.pdgId());
                pt.push_back(c.pt()); eta.push_back(c.eta()); phi.push_back(c.phi()); dr.push_back(DR);
                n = pt.size();
            }
        } muonTPS_, gen_;
        struct Tracks {
            int n;
            std::vector<float> pt, eta, phi;
            void makeBranches(const std::string &prefix, TTree *tree) {
                tree->Branch(("n_"+prefix).c_str(),   &n, (prefix+"/I").c_str());
                tree->Branch((prefix+"_pt").c_str(),  &pt );
                tree->Branch((prefix+"_eta").c_str(), &eta);
                tree->Branch((prefix+"_phi").c_str(), &phi);
             }
            void clear() {
                pt.clear(); eta.clear(); phi.clear();
            }
            void fill(const auto &c) {
                pt.push_back(c.momentum().perp()); eta.push_back(c.eta()); phi.push_back(c.phi());
                n = pt.size();
            }
        } tracks_;

        // input tags  
        edm::EDGetTokenT<std::vector<l1t::TkMuon>> tkMuonSrc_;
        edm::EDGetTokenT<std::vector<reco::GenParticle>> genSrc_;
        edm::EDGetTokenT<std::vector<TTTrack<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>,Phase2TrackerDigi,edm::refhelper::FindForDetSetVector<Phase2TrackerDigi>>>>> tracksSrc_;
        double minPtRatio_;
};

L1TTPSAnalyzer::L1TTPSAnalyzer(const edm::ParameterSet& iConfig):
    tkMuonSrc_(consumes<std::vector<l1t::TkMuon>>(iConfig.getParameter<edm::InputTag>("tkMuonSrc"))),
    genSrc_(consumes<std::vector<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("genSrc"))),
    tracksSrc_(consumes<std::vector<TTTrack<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>,Phase2TrackerDigi,edm::refhelper::FindForDetSetVector<Phase2TrackerDigi>>>>>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
    minPtRatio_(iConfig.getParameter<double>("minRecoPtOverGenPt"))
{
    edm::Service<TFileService> fs;
    tree_ = fs->make<TTree>("tree","tree");
    tree_->Branch("event", &event_, "event/l");
    muonTPS_.makeBranches("muonTPS",tree_);
    gen_.makeBranches("gen",tree_);
    tracks_.makeBranches("track",tree_);
}

L1TTPSAnalyzer::~L1TTPSAnalyzer() { }

void L1TTPSAnalyzer::beginJob() { }
void L1TTPSAnalyzer::endJob() { }

void
L1TTPSAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    muonTPS_.clear();
    gen_.clear();
    tracks_.clear();

    edm::Handle<std::vector<l1t::TkMuon>> tkMuons;
    iEvent.getByToken(tkMuonSrc_, tkMuons);
    edm::Handle<std::vector<reco::GenParticle>> genParticles;
    iEvent.getByToken(genSrc_, genParticles);
    edm::Handle<std::vector<TTTrack<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>,Phase2TrackerDigi,edm::refhelper::FindForDetSetVector<Phase2TrackerDigi>>>>> tracks;
    iEvent.getByToken(tracksSrc_, tracks);

    event_ = iEvent.id().event();
    bool filledMuons = false;
    bool filledTracks = false;
    for (const reco::GenParticle &genParticle : *genParticles) {
        float dr2min = 9801.0;
        //cout << "pt = " << genParticle.pt() << ", eta = " << genParticle.eta() << ", phi = " << genParticle.phi() << ", status = " << genParticle.status() << ", finalState = " << genParticle.isPromptFinalState() << endl;
        if(fabs(genParticle.pdgId())!=13 || genParticle.status()!=1) continue;
        //for (const l1t::TkMuon &tkMuon : *tkMuons) {
        //    if(!filledMuons) muonTPS_.fill(tkMuon,0.0);
        //    if (tkMuon.pt() <= minPtRatio_*genParticle.pt()) continue;
        //    float dr2 = deltaR2(tkMuon.eta(), tkMuon.phi(), genParticle.eta(), genParticle.phi());
        //    if (dr2 < dr2min) dr2min = dr2;
        //}
        //filledMuons = true;
        for (const TTTrack<edm::Ref<edm::DetSetVector<Phase2TrackerDigi>,Phase2TrackerDigi,edm::refhelper::FindForDetSetVector<Phase2TrackerDigi>>> &track : *tracks) {
            if(!filledTracks) tracks_.fill(track);
            float dr2 = deltaR2(track.eta(), track.phi(), genParticle.eta(), genParticle.phi());
            if (dr2 < dr2min) dr2min = dr2;
        }
        filledTracks = true;
        gen_.fill(genParticle,sqrt(dr2min));
    }
    tree_->Fill();
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TTPSAnalyzer);
