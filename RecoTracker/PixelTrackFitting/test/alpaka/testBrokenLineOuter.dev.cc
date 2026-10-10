// Unit test of brokenline::circleFitOuter and brokenline::lineFitOuter
//
// Built for the serial backend only: the kernel body generates the synthetic helices with std:: containers and
// random engines, so it runs as host code. The functions under test are pure arithmetic templates on TAcc.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace bl = ALPAKA_ACCELERATOR_NAMESPACE::brokenline;

namespace {

  constexpr double kBField = 0.0113921;  // GeV/cm/c, 3.8 T

  // Finite-difference steps: 1 um on the interceptions; 1e-3 of the pre-fitted curvature on u_n, large enough that
  // the rounding of phi (~1e-15 rad) is negligible against the change it produces, small enough that the
  // truncation error of the asin term stays below 1e-7.
  constexpr double kStepU = 1e-4;         // cm
  constexpr double kStepKappaRel = 1e-3;  // relative to 1/R
  constexpr double kTolFD = 1e-6;
  // Noiseless helices: the hits are exact to double precision, so the fast fit is the true circle, every fitted
  // offset is zero up to rounding, and the outer state must reproduce the truth up to rounding amplified by the
  // three-point fast fit. 1e-8 is far above that and far below any detector scale (hit resolution >= 1e-3 cm,
  // phi resolution >= 1e-5 rad).
  constexpr double kTolNoiseless = 1e-8;
  // The opposite bending sign moves phi_e by 2 asin(kappa L / 2) >= L / R: >= 6.8e-4 rad for the shortest chord
  // (1.2 cm) at the largest radius (20 GeV) used here.
  constexpr double kMinWrongSignShift = 1e-5;

  struct CaseResult {
    char label[192];
    bool noiseless;
    int q, qFit;
    int expectedM, mCircle, mLine;
    bool expectFallback, fallbackCircle, fallbackLine;
    double chord;
    double relJCircle[3];  // max relative FD deviation per column (u_m, u_{n-1}, u_n), non-zero elements
    bool zeroElementsExact;
    double relCovCircle;
    double relJLine[2];  // max relative FD deviation per column (u_m, u_{n-1}), above the FD resolution
    double relCovLine;
    double maxAbsUnresolvedLine;  // max |analytic - FD| over line elements below the FD resolution
    bool lineWithinResolution;    // every line element agrees to kTolFD relative or within the FD resolution
    bool extractionMatchesCircle, extractionMatchesLine;
    double rhatBU;
    double nlPhi, nlPos, nlKappaRel, nlCot, nlZ, wrongSignShift, nlCotUpstream;
    double upCot, upZ;  // |cot_out - cot_upstream| and |z_e - (zip_upstream + cot_upstream * s_e)|
    bool dataUnchanged, upstreamIdentical, kappaIdentical;
    bool orderDependent;
    double orderDPhi, orderDZ;
  };

  double relErr(double a, double b, double floor) {
    return std::abs(a - b) / std::max({std::abs(a), std::abs(b), floor, std::numeric_limits<double>::min()});
  }

  double wrapPhi(double d) { return std::remainder(d, 2 * M_PI); }

  // helix at transverse arc length s; q > 0 bends clockwise for Bz > 0
  void helixAt(double s,
               double radius,
               int q,
               double phi0,
               double cot,
               double x0,
               double y0,
               double z0,
               double& x,
               double& y,
               double& z,
               double& phi) {
    phi = phi0 - q * s / radius;
    x = x0 + (radius / q) * (std::sin(phi0) - std::sin(phi));
    y = y0 + (radius / q) * (std::cos(phi) - std::cos(phi0));
    z = z0 + cot * s;
  }

  template <int N>
  bool sameData(const bl::PreparedBrokenLineData<N>& a, const bl::PreparedBrokenLineData<N>& b) {
    return a.qCharge == b.qCharge && !std::memcmp(a.radii.data(), b.radii.data(), sizeof(double) * 2 * N) &&
           !std::memcmp(a.sTransverse.data(), b.sTransverse.data(), sizeof(double) * N) &&
           !std::memcmp(a.sTotal.data(), b.sTotal.data(), sizeof(double) * N) &&
           !std::memcmp(a.zInSZplane.data(), b.zInSZplane.data(), sizeof(double) * N) &&
           !std::memcmp(a.varBeta.data(), b.varBeta.data(), sizeof(double) * N);
  }

  template <int N>
  __attribute__((noinline)) void upstreamFits(const Acc1D& acc,
                                              const ::riemannFit::Matrix3xNd<N>& hits,
                                              const Eigen::Matrix<float, 6, N>& hits_ge,
                                              const ::riemannFit::Vector4d& fast_fit,
                                              bl::PreparedBrokenLineData<N>& data,
                                              bl::karimaki_circle_fit& circle,
                                              ::riemannFit::LineFit& line) {
    bl::lineFit(acc, hits_ge, fast_fit, kBField, data, line);
    bl::circleFit(acc, hits, hits_ge, fast_fit, kBField, data, circle);
  }

  template <int N>
  CaseResult runCase(const Acc1D& acc,
                     const char* layout,
                     const std::vector<double>& radii,
                     int expectedM,
                     bool expectFallback,
                     double pt,
                     int q,
                     double phi0,
                     double cot,
                     bool noiseless,
                     std::mt19937& rng) {
    CaseResult res{};
    std::snprintf(res.label,
                  sizeof(res.label),
                  "layout=%s n=%d pt=%.1f q=%+d phi0=%.2f cot=%.2f %s",
                  layout,
                  N,
                  pt,
                  q,
                  phi0,
                  cot,
                  noiseless ? "noiseless" : "smeared");
    res.noiseless = noiseless;
    res.q = q;
    res.expectedM = expectedM;
    res.expectFallback = expectFallback;

    // ---- synthetic helix and hits
    const double radius = pt / kBField;
    const double d0 = 0.01, z0 = 0.5;
    const double x0 = -d0 * std::sin(phi0), y0 = d0 * std::cos(phi0);
    std::normal_distribution<double> gaus(0., 1.);
    ::riemannFit::Matrix3xNd<N> hits;
    Eigen::Matrix<float, 6, N> hits_ge;
    double trueX = 0, trueY = 0, trueZ = 0, truePhi = 0;
    double sPrev = 0;
    for (int i = 0; i < N; ++i) {
      double lo = sPrev, hi = M_PI * radius, x, y, z, ph;
      for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (lo + hi);
        helixAt(mid, radius, q, phi0, cot, x0, y0, z0, x, y, z, ph);
        (std::hypot(x, y) < radii[i] ? lo : hi) = mid;
      }
      helixAt(lo, radius, q, phi0, cot, x0, y0, z0, x, y, z, ph);
      sPrev = lo;
      const double r = std::hypot(x, y);
      const double sigRPhi = r < 20 ? 0.001 : 0.005, sigR = 1e-4, sigZ = r < 20 ? 0.002 : 0.1;
      const double tx = -y / r, ty = x / r, rx = x / r, ry = y / r;
      if (i == N - 1) {
        trueX = x;
        trueY = y;
        trueZ = z;
        truePhi = ph;
      }
      if (!noiseless) {
        const double a = sigRPhi * gaus(rng), b = sigR * gaus(rng);
        x += a * tx + b * rx;
        y += a * ty + b * ry;
        z += sigZ * gaus(rng);
      }
      hits(0, i) = x;
      hits(1, i) = y;
      hits(2, i) = z;
      hits_ge(0, i) = sigRPhi * sigRPhi * tx * tx + sigR * sigR * rx * rx;
      hits_ge(1, i) = sigRPhi * sigRPhi * tx * ty + sigR * sigR * rx * ry;
      hits_ge(2, i) = sigRPhi * sigRPhi * ty * ty + sigR * sigR * ry * ry;
      hits_ge(3, i) = 0;
      hits_ge(4, i) = 0;
      hits_ge(5, i) = sigZ * sigZ;
    }

    ::riemannFit::Vector4d fast_fit;
    bl::fastFit(acc, hits, fast_fit);
    bl::PreparedBrokenLineData<N> data;
    bl::prepareBrokenLineData(acc, hits, fast_fit, kBField, data);
    const bl::PreparedBrokenLineData<N> dataOrig = data;
    bl::PreparedBrokenLineData<N> dataFresh = data;

    // ---- the new functions first; data must not change
    bl::OuterCircleState oc;
    bl::OuterLineState ol;
    bl::circleFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, oc);
    bl::lineFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, ol);
    res.dataUnchanged = sameData(data, dataOrig);
    res.qFit = oc.qCharge;
    res.mCircle = oc.mRef;
    res.mLine = ol.mRef;
    res.fallbackCircle = oc.refFallback;
    res.fallbackLine = ol.refFallback;

    // ---- upstream after the new functions (on data) vs upstream alone (on dataFresh), in the LST order
    bl::karimaki_circle_fit c1, c2;
    ::riemannFit::LineFit l1, l2;
    upstreamFits(acc, hits, hits_ge, fast_fit, data, c1, l1);
    upstreamFits(acc, hits, hits_ge, fast_fit, dataFresh, c2, l2);
    res.upstreamIdentical = !std::memcmp(c1.par.data(), c2.par.data(), sizeof(double) * 3) &&
                            !std::memcmp(c1.cov.data(), c2.cov.data(), sizeof(double) * 9) &&
                            !std::memcmp(&c1.chi2, &c2.chi2, sizeof(c1.chi2)) && c1.qCharge == c2.qCharge &&
                            !std::memcmp(l1.par.data(), l2.par.data(), sizeof(double) * 2) &&
                            !std::memcmp(l1.cov.data(), l2.cov.data(), sizeof(double) * 4) &&
                            !std::memcmp(&l1.chi2, &l2.chi2, sizeof(l1.chi2));
    res.kappaIdentical = !std::memcmp(&oc.par(2), &c2.par(2), sizeof(double));

    // ---- information only: the precondition, i.e. the new functions called AFTER upstream circleFit
    {
      bl::OuterCircleState oc2;
      bl::OuterLineState ol2;
      bl::circleFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, oc2);
      bl::lineFitOuter(acc, hits, hits_ge, fast_fit, kBField, data, ol2);
      res.orderDependent = oc2.par != oc.par || ol2.par != ol.par || oc2.cov != oc.cov || ol2.cov != ol.cov;
      res.orderDPhi = std::abs(wrapPhi(oc2.par(0) - oc.par(0)));
      res.orderDZ = std::abs(ol2.par(1) - ol.par(1));
    }

    const int m = oc.mRef;

    // ---- the circle solve of circleFit(), repeated here to obtain iMat and the fitted parameters
    bl::PreparedBrokenLineData<N> d = dataOrig;
    const double slope = -d.qCharge / fast_fit(3);
    d.varBeta *= 1. + ::riemannFit::sqr(slope);
    for (int i = 0; i < N; i++)
      d.zInSZplane(i) = d.radii.block(0, i, 2, 1).norm() - fast_fit(2);
    ::riemannFit::VectorNd<N> weights;
    for (int i = 0; i < N; i++) {
      ::riemannFit::Matrix2d vMat;
      vMat(0, 0) = hits_ge.col(i)[0];
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];
      vMat(1, 1) = hits_ge.col(i)[2];
      ::riemannFit::Matrix2d rotMat = bl::rotationMatrix(acc, -d.radii(0, i) / d.radii(1, i));
      weights(i) = 1. / ((rotMat * vMat * rotMat.transpose())(1, 1));
    }
    ::riemannFit::VectorNplusONEd<N> r_u;
    r_u(N) = 0;
    for (int i = 0; i < N; i++)
      r_u(i) = weights(i) * d.zInSZplane(i);
    const auto& sT = d.sTransverse;
    const auto& vB = d.varBeta;
    ::riemannFit::MatrixNplusONEd<N> c_u;
    c_u.block(0, 0, N, N) = bl::matrixC_u(acc, weights, sT, vB);
    c_u(N, N) = 0;
    for (int i = 0; i < N; i++) {
      c_u(i, N) = 0;
      if (i > 0 && i < N - 1)
        c_u(i, N) += -(sT(i + 1) - sT(i - 1)) * (sT(i + 1) - sT(i - 1)) /
                     (2. * vB(i) * (sT(i + 1) - sT(i)) * (sT(i) - sT(i - 1)));
      if (i > 1)
        c_u(i, N) += (sT(i) - sT(i - 2)) / (2. * vB(i - 1) * (sT(i) - sT(i - 1)));
      if (i < N - 2)
        c_u(i, N) += (sT(i + 2) - sT(i)) / (2. * vB(i + 1) * (sT(i + 1) - sT(i)));
      c_u(N, i) = c_u(i, N);
      if (i > 0 && i < N - 1)
        c_u(N, N) += ::riemannFit::sqr(sT(i + 1) - sT(i - 1)) / (4. * vB(i));
    }
    ::riemannFit::MatrixNplusONEd<N> iMat;
    math::cholesky::invert(c_u, iMat);
    const ::riemannFit::VectorNplusONEd<N> u = iMat * r_u;

    const ::riemannFit::Vector2d hitA = hits.block(0, m, 2, 1), hitB = hits.block(0, N - 1, 2, 1);
    const ::riemannFit::Vector2d rhatA = d.radii.col(m) / d.radii.col(m).norm();
    const ::riemannFit::Vector2d rhatB = d.radii.col(N - 1) / d.radii.col(N - 1).norm();
    const double zA = d.zInSZplane(m), zB = d.zInSZplane(N - 1);
    auto extract = [&](double uA, double uB, double uN, bl::OuterCircleState& s, ::riemannFit::Matrix3d& jac) {
      s.qCharge = d.qCharge;
      bl::outerCircleExtraction(acc, hitA, hitB, rhatA, rhatB, uA, zA, uB, zB, uN, fast_fit(2), s, jac);
    };

    bl::OuterCircleState s0;
    ::riemannFit::Matrix3d jAn;
    extract(u(m), u(N - 1), u(N), s0, jAn);
    res.extractionMatchesCircle = s0.par == oc.par && s0.point == oc.point;
    const ::riemannFit::Vector2d chordVec = s0.point - (hitA + (u(m) - zA) * rhatA);
    res.chord = chordVec.norm();
    res.rhatBU = jAn(1, 1);

    // ---- circle Jacobian: central finite differences in (u_m, u_{n-1}, u_n)
    const ::riemannFit::Vector2d normalU(-std::sin(s0.par(0)), std::cos(s0.par(0)));
    ::riemannFit::Matrix3d jFD;
    for (int k = 0; k < 3; ++k) {
      double up[3] = {u(m), u(N - 1), u(N)}, dn[3] = {u(m), u(N - 1), u(N)};
      const double h = k == 2 ? kStepKappaRel / fast_fit(2) : kStepU;
      up[k] += h;
      dn[k] -= h;
      bl::OuterCircleState sp, sm;
      ::riemannFit::Matrix3d jUnused;
      extract(up[0], up[1], up[2], sp, jUnused);
      extract(dn[0], dn[1], dn[2], sm, jUnused);
      jFD(0, k) = wrapPhi(sp.par(0) - sm.par(0)) / (2 * h);
      jFD(1, k) = (sp.point - sm.point).dot(normalU) / (2 * h);
      jFD(2, k) = (sp.par(2) - sm.par(2)) / (2 * h);
    }
    res.zeroElementsExact = true;
    for (int k = 0; k < 3; ++k) {
      res.relJCircle[k] = 0;
      for (int i = 0; i < 3; ++i) {
        const bool analyticZero = (i == 1 && k != 1) || (i == 2 && k != 2);
        if (analyticZero)
          res.zeroElementsExact = res.zeroElementsExact && jAn(i, k) == 0 && jFD(i, k) == 0;
        else
          res.relJCircle[k] = std::max(res.relJCircle[k], relErr(jAn(i, k), jFD(i, k), 0.));
      }
    }

    // ---- circle covariance: J_FD * iMat({m, n-1, n}) * J_FD^T plus the chord-wide m.s. term on phi_e
    ::riemannFit::Matrix3d iBlock;
    const int idx[3] = {m, N - 1, N};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        iBlock(i, k) = iMat(idx[i], idx[k]);
    ::riemannFit::Matrix3d covExp = jFD * iBlock * jFD.transpose();
    covExp(0, 0) +=
        (1 + slope * slope) * bl::multScatt(acc, d.sTotal(N - 1) - d.sTotal(m), kBField, fast_fit(2), N, slope);
    res.relCovCircle = 0;
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k)
        res.relCovCircle = std::max(res.relCovCircle,
                                    relErr(oc.cov(i, k), covExp(i, k), 1e-9 * std::sqrt(covExp(i, i) * covExp(k, k))));

    // ---- the line solve of lineFit(), repeated here to obtain iMat and the fitted parameters
    const ::riemannFit::Matrix2d rotMatL = bl::rotationMatrix(acc, slope);
    ::riemannFit::VectorNd<N> weightsL;
    for (int i = 0; i < N; i++) {
      ::riemannFit::Matrix3d vMat = ::riemannFit::Matrix3d::Zero();
      ::riemannFit::Matrix2x3d jac = ::riemannFit::Matrix2x3d::Zero();
      vMat(0, 0) = hits_ge.col(i)[0];
      vMat(0, 1) = vMat(1, 0) = hits_ge.col(i)[1];
      vMat(0, 2) = vMat(2, 0) = hits_ge.col(i)[3];
      vMat(1, 1) = hits_ge.col(i)[2];
      vMat(2, 1) = vMat(1, 2) = hits_ge.col(i)[4];
      vMat(2, 2) = hits_ge.col(i)[5];
      const double tmp = 1. / dataOrig.radii.block(0, i, 2, 1).norm();
      jac(0, 0) = dataOrig.qCharge * dataOrig.radii(1, i) * tmp;
      jac(0, 1) = -dataOrig.qCharge * dataOrig.radii(0, i) * tmp;
      jac(1, 2) = 1.;
      weightsL(i) = 1. / ((rotMatL * jac * vMat * jac.transpose() * rotMatL.transpose())(1, 1));
    }
    ::riemannFit::VectorNd<N> r_uL;
    for (int i = 0; i < N; i++)
      r_uL(i) = weightsL(i) * dataOrig.zInSZplane(i);
    ::riemannFit::MatrixNd<N> iMatL;
    math::cholesky::invert(bl::matrixC_u(acc, weightsL, dataOrig.sTotal, dataOrig.varBeta), iMatL);
    const ::riemannFit::VectorNd<N> uL = iMatL * r_uL;
    ::riemannFit::Matrix2d iBlockL;
    iBlockL << iMatL(m, m), iMatL(m, N - 1), iMatL(N - 1, m), iMatL(N - 1, N - 1);
    const double sA = dataOrig.sTotal(m), sB = dataOrig.sTotal(N - 1), sE = dataOrig.sTransverse(N - 1);
    const double msVar = bl::multScatt(acc, sB - sA, kBField, fast_fit(2), N, slope);
    auto extractLine = [&](double uA, double uB, bl::OuterLineState& s) {
      bl::outerLineExtraction(acc, uA, uB, iBlockL, sA, sB, sE, rotMatL, msVar, s);
    };
    bl::OuterLineState l0;
    extractLine(uL(m), uL(N - 1), l0);
    res.extractionMatchesLine = l0.par == ol.par && l0.cov == ol.cov;

    // ---- line Jacobian: central finite differences in (u_m, u_{n-1}) vs the chain of the three linear steps
    ::riemannFit::Matrix2d jFDL;
    for (int k = 0; k < 2; ++k) {
      double up[2] = {uL(m), uL(N - 1)}, dn[2] = {uL(m), uL(N - 1)};
      up[k] += kStepU;
      dn[k] -= kStepU;
      bl::OuterLineState sp, sm;
      extractLine(up[0], up[1], sp);
      extractLine(dn[0], dn[1], sm);
      jFDL.col(k) = (sp.par - sm.par) / (2 * kStepU);
    }
    const double chordS = sB - sA;
    const double slopeRot = (uL(N - 1) - uL(m)) / chordS, zipRot = uL(N - 1) - slopeRot * sB;
    const double tmpRot = rotMatL(0, 0) - slopeRot * rotMatL(0, 1);
    ::riemannFit::Matrix2d jPair, jTrans, jRot, jShift;
    jPair << -1. / chordS, 1. / chordS, 0, 1;
    jTrans << 1, 0, -sB, 1;
    jRot << 1. / (tmpRot * tmpRot), 0, zipRot * rotMatL(0, 1) / (tmpRot * tmpRot), 1. / tmpRot;
    jShift << 1, 0, sE, 1;
    const ::riemannFit::Matrix2d jAnL = jShift * jRot * jTrans * jPair;
    // FD resolution of the line map: (cot, z_e) are evaluated with a rounding of a few ulp of the larger of the two
    // (|z_e| up to ~2e2 cm), so a central difference cannot resolve derivatives below ~16 eps max|par| / h. The
    // element dz_e/du_m falls below that: it is proportional to the residual at node n-1 (exactly 0 for exact hits),
    // because z_e is taken at that node. Such elements are compared at the resolution, all others at kTolFD.
    const double resolutionL = 16 * std::numeric_limits<double>::epsilon() * ol.par.cwiseAbs().maxCoeff() / kStepU;
    res.lineWithinResolution = true;
    res.maxAbsUnresolvedLine = 0;
    for (int k = 0; k < 2; ++k) {
      res.relJLine[k] = 0;
      for (int i = 0; i < 2; ++i) {
        if (std::max(std::abs(jAnL(i, k)), std::abs(jFDL(i, k))) * kTolFD > resolutionL) {
          res.relJLine[k] = std::max(res.relJLine[k], relErr(jAnL(i, k), jFDL(i, k), 0.));
        } else {
          const double deviation = std::abs(jAnL(i, k) - jFDL(i, k));
          res.maxAbsUnresolvedLine = std::max(res.maxAbsUnresolvedLine, deviation);
          res.lineWithinResolution = res.lineWithinResolution && deviation < resolutionL;
        }
      }
    }
    // the m.s. variance sits on the rotated slope at fixed u_{n-1}: d/d(slope) = -(S_b - S_a) d/du_m
    const ::riemannFit::Matrix2d covExpL =
        jFDL * iBlockL * jFDL.transpose() + msVar * chordS * chordS * jFDL.col(0) * jFDL.col(0).transpose();
    res.relCovLine = 0;
    for (int i = 0; i < 2; ++i)
      for (int k = 0; k < 2; ++k)
        res.relCovLine = std::max(res.relCovLine, relErr(ol.cov(i, k), covExpL(i, k), 0.));

    // ---- truth at the last hit
    res.nlPhi = std::abs(wrapPhi(oc.par(0) - truePhi));
    res.nlPos = std::hypot(oc.point(0) - trueX, oc.point(1) - trueY);
    res.nlKappaRel = std::abs(oc.par(2) * radius - q);
    res.nlCot = std::abs(ol.par(0) - cot);
    res.nlCotUpstream = std::abs(l2.par(0) - cot);  // the same quantity from upstream lineFit, for comparison
    res.nlZ = std::abs(ol.par(1) - trueZ);
    // ---- agreement with upstream lineFit on the same inputs (l2, from upstreamFits on dataFresh). lineFit already
    // returns par = (cot(theta), Zip) in the original sz system (rotated back as at its "rotate to the original sz
    // system" step), so the upstream z at the outermost node is Zip + cot(theta) * s_e, with s_e = sTransverse(n-1)
    // of the same prepared data, exactly the last step of outerLineExtraction().
    const double sOuter = dataOrig.sTransverse(N - 1);
    res.upCot = std::abs(ol.par(0) - l2.par(0));
    res.upZ = std::abs(ol.par(1) - (l2.par(1) + l2.par(0) * sOuter));
    // the tangent with the opposite bending sign
    const double phiWrong = std::atan2(chordVec(1), chordVec(0)) + std::asin(0.5 * oc.par(2) * chordVec.norm());
    res.wrongSignShift = std::abs(wrapPhi(phiWrong - truePhi));
    return res;
  }

  struct OuterFitTestKernel {
    ALPAKA_FN_ACC void operator()(const Acc1D& acc, std::vector<CaseResult>* results) const {
      std::mt19937 rng(12345);
      // transverse radii of the hits (cm); expected m from the 2 cm rule
      const std::vector<double> layoutA = {23, 23.16, 36, 36.16, 51, 51.16, 68, 68.16, 87, 87.16};
      const std::vector<double> layoutB = {3, 7, 11, 16, 23, 23.16, 36, 36.16, 51, 51.16, 68, 68.16};
      const std::vector<double> layoutC = {23, 23.16, 30, 30.16, 32.3, 32.46};
      const std::vector<double> layoutD = {23, 23.16, 51, 51.16, 81, 81.16};
      const std::vector<double> layoutE = {23, 23.16, 36, 36.16, 51, 51.16, 68};
      const std::vector<double> layoutF = {40, 40.16, 40.5, 40.66, 41.2};
      const std::vector<double> layoutG = {3, 7, 11, 16};
      const double pts[4] = {0.8, 2., 5., 20.};
      const double phis[3] = {0.3, 3.1, -1.7};
      const double cots[3] = {0.3, -1.2, 2.5};
      for (bool noiseless : {true, false})
        for (int q : {+1, -1})
          for (double pt : pts)
            for (int a = 0; a < 3; ++a) {
              auto& r = *results;
              r.push_back(runCase<10>(acc, "OT pairs", layoutA, 7, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<12>(acc, "pixel+OT", layoutB, 9, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<6>(acc, "chord 2.3", layoutC, 3, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<6>(acc, "chord 30", layoutD, 3, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<7>(acc, "single last", layoutE, 5, false, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<5>(acc, "fallback", layoutF, 0, true, pt, q, phis[a], cots[a], noiseless, rng));
              r.push_back(runCase<4>(acc, "pixel quad", layoutG, 2, false, pt, q, phis[a], cots[a], noiseless, rng));
            }
    }
  };

}  // namespace

TEST_CASE("Broken Line state at the outermost node: " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE), "[BrokenLine]") {
  auto const& devices = cms::alpakatools::devices<Platform>();
  REQUIRE(not devices.empty());
  Queue queue(devices.front());
  std::vector<CaseResult> results;
  alpaka::exec<Acc1D>(queue, cms::alpakatools::make_workdiv<Acc1D>(1, 1), OuterFitTestKernel{}, &results);
  alpaka::wait(queue);
  REQUIRE(results.size() == 336);

  double maxJCircle[3] = {0, 0, 0}, maxJLine[2] = {0, 0}, maxCovCircle = 0, maxCovLine = 0;
  double maxNlPhi = 0, maxNlPos = 0, maxNlKappa = 0, minWrongShift = 1e9;
  double minChord = 1e9, maxChord = 0, minQRhatBU = 1e9, maxOrderDPhi = 0, maxOrderDZ = 0, maxUnresolvedLine = 0;
  // index 0: all layouts but the fallback one; index 1: the fallback layout
  double maxNlCot[2] = {0, 0}, maxNlZ[2] = {0, 0}, maxNlCotUpstream[2] = {0, 0}, maxUpCot[2] = {0, 0},
         maxUpZ[2] = {0, 0};
  int nFallback = 0, nOrderDependent = 0, nChargeFlips = 0;

  for (auto const& r : results) {
    INFO(r.label);
    // With smeared hits the generated charge is not always recoverable: over the 1.2 cm "fallback" span and the
    // 2.3 cm chord at 20 GeV the sagitta is below the 50 um hit resolution. The fit's charge is the three-point sign
    // of prepareBrokenLineData(), and the sign conventions below are checked against it.
    if (r.noiseless) {
      INFO("fitted charge differs from the generated charge for exact hits");
      CHECK(r.qFit == r.q);
    }
    {
      INFO("reference node m differs from the 2 cm rule expectation");
      CHECK(r.mCircle == r.expectedM);
    }
    {
      INFO("line and circle use different reference nodes");
      CHECK(r.mLine == r.mCircle);
    }
    {
      INFO("unexpected fallback flag (circle)");
      CHECK(r.fallbackCircle == r.expectFallback);
    }
    {
      INFO("unexpected fallback flag (line)");
      CHECK(r.fallbackLine == r.expectFallback);
    }
    {
      INFO("outerCircleExtraction at the solution differs from circleFitOuter");
      CHECK(r.extractionMatchesCircle);
    }
    {
      INFO("outerLineExtraction at the solution differs from lineFitOuter");
      CHECK(r.extractionMatchesLine);
    }
    for (int k = 0; k < 3; ++k) {
      INFO("circle Jacobian column " << k << " (0: u_m, 1: u_{n-1}, 2: u_n), rel. deviation " << r.relJCircle[k]);
      {
        INFO("circle Jacobian disagrees with finite differences");
        CHECK(r.relJCircle[k] < kTolFD);
      }
    }
    {
      INFO("circle Jacobian elements that are analytically zero are not zero");
      CHECK(r.zeroElementsExact);
    }
    INFO("circle covariance rel. deviation " << r.relCovCircle);
    {
      INFO("circle covariance disagrees with J_FD * iMat * J_FD^T + m.s.");
      CHECK(r.relCovCircle < kTolFD);
    }
    for (int k = 0; k < 2; ++k) {
      INFO("line Jacobian column " << k << " (0: u_m, 1: u_{n-1}), rel. deviation " << r.relJLine[k]);
      {
        INFO("line Jacobian disagrees with finite differences");
        CHECK(r.relJLine[k] < kTolFD);
      }
    }
    {
      INFO("line Jacobian element below the FD resolution disagrees beyond it, |deviation| " << r.maxAbsUnresolvedLine);
      CHECK(r.lineWithinResolution);
    }
    INFO("line covariance rel. deviation " << r.relCovLine);
    {
      INFO("line covariance disagrees with J_FD * iMat * J_FD^T + m.s.");
      CHECK(r.relCovLine < kTolFD);
    }
    INFO("rhat_b . U = " << r.rhatBU);
    {
      INFO("d x_T / d u_b does not have the sign of the fitted charge");
      CHECK(r.rhatBU * r.qFit > 0);
    }
    {
      INFO("the new functions modified PreparedBrokenLineData");
      CHECK(r.dataUnchanged);
    }
    {
      INFO("upstream circleFit/lineFit outputs changed by calling the new functions first");
      CHECK(r.upstreamIdentical);
    }
    {
      INFO("circleFitOuter kappa is not bit-identical to circleFit kappa");
      CHECK(r.kappaIdentical);
    }
    if (r.noiseless) {
      INFO("noiseless residuals: phi " << r.nlPhi << " pos " << r.nlPos << " kappa(rel) " << r.nlKappaRel << " cot "
                                       << r.nlCot << " z " << r.nlZ << "; opposite-sign shift " << r.wrongSignShift);
      {
        INFO("phi_e differs from the true tangent at the last hit");
        CHECK(r.nlPhi < kTolNoiseless);
      }
      {
        INFO("eB differs from the true last hit");
        CHECK(r.nlPos < kTolNoiseless);
      }
      {
        INFO("kappa differs from the true curvature");
        CHECK(r.nlKappaRel < kTolNoiseless);
      }
      // The line truth checks exclude the fallback layout (1.2 cm span): there the three-point fast fit is
      // ill-conditioned by construction, and upstream lineFit misses the true cot(theta) by the same amount.
      if (!r.expectFallback) {
        {
          INFO("cot_out differs from the true cot(theta)");
          CHECK(r.nlCot < kTolNoiseless);
        }
        {
          INFO("z_e differs from the true z at the last hit");
          CHECK(r.nlZ < kTolNoiseless);
        }
      }
      // On every layout the outer line must agree with upstream lineFit on the same inputs.
      INFO("upstream agreement: |d cot| " << r.upCot << " |d z_e| " << r.upZ);
      {
        INFO("cot_out differs from upstream lineFit cot(theta)");
        CHECK(r.upCot < kTolNoiseless);
      }
      {
        INFO("z_e differs from upstream lineFit Zip + cot(theta) * sTransverse(n-1)");
        CHECK(r.upZ < kTolNoiseless);
      }
      {
        INFO("the opposite bending sign also matches the truth");
        CHECK(r.wrongSignShift > kMinWrongSignShift);
      }
      maxNlPhi = std::max(maxNlPhi, r.nlPhi);
      maxNlPos = std::max(maxNlPos, r.nlPos);
      maxNlKappa = std::max(maxNlKappa, r.nlKappaRel);
      const int fb = r.expectFallback ? 1 : 0;  // 1 for the fallback layout
      maxNlCot[fb] = std::max(maxNlCot[fb], r.nlCot);
      maxNlCotUpstream[fb] = std::max(maxNlCotUpstream[fb], r.nlCotUpstream);
      maxNlZ[fb] = std::max(maxNlZ[fb], r.nlZ);
      maxUpCot[fb] = std::max(maxUpCot[fb], r.upCot);
      maxUpZ[fb] = std::max(maxUpZ[fb], r.upZ);
      minWrongShift = std::min(minWrongShift, r.wrongSignShift);
    }
    for (int k = 0; k < 3; ++k)
      maxJCircle[k] = std::max(maxJCircle[k], r.relJCircle[k]);
    for (int k = 0; k < 2; ++k)
      maxJLine[k] = std::max(maxJLine[k], r.relJLine[k]);
    maxCovCircle = std::max(maxCovCircle, r.relCovCircle);
    maxCovLine = std::max(maxCovLine, r.relCovLine);
    minChord = std::min(minChord, r.chord);
    maxChord = std::max(maxChord, r.chord);
    minQRhatBU = std::min(minQRhatBU, r.qFit * r.rhatBU);
    maxUnresolvedLine = std::max(maxUnresolvedLine, r.maxAbsUnresolvedLine);
    nChargeFlips += (r.qFit != r.q);
    nFallback += r.fallbackCircle;
    nOrderDependent += r.orderDependent;
    maxOrderDPhi = std::max(maxOrderDPhi, r.orderDPhi);
    maxOrderDZ = std::max(maxOrderDZ, r.orderDZ);
  }

  std::cout << "Summary over " << results.size() << " cases\n"
            << "  max rel. FD deviation, circle Jacobian columns (u_m, u_{n-1}, u_n): " << maxJCircle[0] << ' '
            << maxJCircle[1] << ' ' << maxJCircle[2] << '\n'
            << "  max rel. FD deviation, line Jacobian columns (u_m, u_{n-1}): " << maxJLine[0] << ' ' << maxJLine[1]
            << "; max |deviation| of elements below the FD resolution " << maxUnresolvedLine << '\n'
            << "  max rel. deviation of the covariances, circle / line: " << maxCovCircle << " / " << maxCovLine << '\n'
            << "  noiseless max |phi_e - true| " << maxNlPhi << " rad, |eB - true| " << maxNlPos
            << " cm, |kappa R / q - 1| " << maxNlKappa << '\n'
            << "  noiseless, non-fallback layouts: max |cot_out - true| " << maxNlCot[0] << ", |z_e - true| "
            << maxNlZ[0] << " cm, upstream lineFit |cot - true| " << maxNlCotUpstream[0] << "; vs upstream |d cot| "
            << maxUpCot[0] << ", |d z_e| " << maxUpZ[0] << " cm\n"
            << "  noiseless, fallback layout (truth not asserted): max |cot_out - true| " << maxNlCot[1]
            << ", |z_e - true| " << maxNlZ[1] << " cm, upstream lineFit |cot - true| " << maxNlCotUpstream[1]
            << "; vs upstream |d cot| " << maxUpCot[1] << ", |d z_e| " << maxUpZ[1] << " cm\n"
            << "  noiseless min shift of the opposite-sign tangent " << minWrongShift << " rad\n"
            << "  min q * (rhat_b . U) " << minQRhatBU << "; chord range " << minChord << " - " << maxChord << " cm\n"
            << "  fallback (m = 0) cases " << nFallback << "; smeared cases with a flipped fitted charge "
            << nChargeFlips << '\n'
            << "  information only: results differ when called after upstream circleFit in " << nOrderDependent
            << " of " << results.size() << " cases (max |d phi_e| " << maxOrderDPhi << " rad, max |d z_e| "
            << maxOrderDZ << " cm)" << std::endl;
}
