// Copyright(C) 2021 Shinsuke Takasao <takasao@astro-osaka.jp>
// Licensed under the 3-clause BSD License, see LICENSE file for details
// This code uses two header files in Athena++, athena_arrays.hpp and defs.hpp:
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//
// Compile command:
// >> g++ -std=c++11 AdvectionSimpleTest.cpp -Wall
//
// Description about the input parameters specified in advinput.txt:
// icmodel : "SineWave" or "ComplexWave"
// integrator : rk2, rk3, LW
// xorder : 1=1st order, 2a=Lax-Wendroff, 2b=minmod, 2c=MC, 2d=superbee,
// 4=4th order ver of WENO-Z (Borges et al. 2008)
//
// The output data can be easily read using athena_read.tab:
// > data = athena_read.tab(filename)

#include <algorithm>
#include <cmath>      // sqrt()
#include <csignal>    // ISO C/C++ signal() and sigset_t, sigemptyset() POSIX C extensions
#include <cstdint>    // int64_t
#include <cstdio>     // sscanf()
#include <cstdlib>    // strtol
#include <ctime>      // clock(), CLOCKS_PER_SEC, clock_t
#include <exception>  // exception
#include <iomanip>    // setprecision()
#include <iostream>   // cout, endl
#include <fstream>    // ofstream
#include <limits>     // max_digits10
#include <new>        // bad_alloc
#include <string>     // string
#include <sstream>    // ostringstream
#include <vector>

// Athena++ headers
#include "athena_arrays.hpp"
#include "defs.hpp"

using Real = double;
#define MAX_NSTAGES 5

std::vector<std::string> split(std::string str, std::string separator) {
    if (separator == "") return {str};
    std::vector<std::string> result;
    std::string tstr = str + separator;
    long l = tstr.length(), sl = separator.length();
    std::string::size_type pos = 0, prev = 0;
    
    for (;pos < l && (pos = tstr.find(separator, pos)) != std::string::npos; prev = (pos += sl)) {
        result.emplace_back(tstr, prev, pos - prev);
    }
    return result;
}

struct IntegratorWeight { 
  Real beta;
};
class Reconstruction {
public:
  Reconstruction(int nc1);
  ~Reconstruction();
  AthenaArray<Real> ul_,ur_;
  AthenaArray<Real> uc_,dul_,dur_,dum_; // for PLM
  AthenaArray<Real> beta0_,beta1_,beta2_,alpha0_,alpha1_,alpha2_
    ,wz0_,wz1_,wz2_,tau5_; // for WENO-Z
};
Reconstruction::Reconstruction(int nc1){
  ul_.NewAthenaArray(nc1);
  ur_.NewAthenaArray(nc1);

  uc_.NewAthenaArray(nc1);
  dul_.NewAthenaArray(nc1);
  dur_.NewAthenaArray(nc1);
  dum_.NewAthenaArray(nc1);

  beta0_.NewAthenaArray(nc1);
  beta1_.NewAthenaArray(nc1);
  beta2_.NewAthenaArray(nc1);
  alpha0_.NewAthenaArray(nc1);
  alpha1_.NewAthenaArray(nc1);
  alpha2_.NewAthenaArray(nc1);
  wz0_.NewAthenaArray(nc1);
  wz1_.NewAthenaArray(nc1);
  wz2_.NewAthenaArray(nc1);
  tau5_.NewAthenaArray(nc1);
}


// declaration
Real func_G(Real x, Real b, Real z);
Real func_F(Real x, Real alp, Real a);
void CalculateFluxes(AthenaArray<Real> &flux, AthenaArray<Real> &u,
                     const Real v, const Real dt, const Real dx, const int is, const int ie, const std::string order,
                     Reconstruction *prcn);
void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u0, const int is, const int ie,
                 const int s, const std::string integrator);
void SetInitialCondition(AthenaArray<Real> &u, AthenaArray<Real> &x,
                         const Real x1min, const Real x1max,
                         const int il, const int iu, std::string icmodel);
void SetBoundaryCondition(AthenaArray<Real> &u, const int il, const int iu, const int ng);

std::string ZeroPadNumber(int num, int numlen);
void WriteData(int &nwrite, AthenaArray<Real> &u, AthenaArray<Real> &x, AthenaArray<Real> &dxv,
               const Real t, const int ncycle, const int ncells1);
void CalculateError(AthenaArray<Real> &u, AthenaArray<Real> &uan, AthenaArray<Real> &x1v,
  AthenaArray<Real> &dx1v, const Real time, const int il, const int iu);

namespace {
  Real cfl, c, tlim;
  std::string integrator, icmodel, xorder, ofilename;
  int nx1;
}

int main() {
  // read input file
  std::ifstream ifs("./advinput.txt");
  if (ifs.fail()) {
    std::cerr << "Failed to open the input file, advinput.txt." << std::endl;
    return -1;
  }
  std::string str;
  while (std::getline(ifs, str)) {
    std::vector<std::string> ary = split(str,"=");
    // remove spaces
    for (int i = 0; i < ary.size(); i++) {
      std::string temp = ary[i];
      temp.erase(remove(temp.begin(), temp.end(),' '), temp.end());
      ary[i] = temp;
    }

    if (ary[0]=="cfl") {
      cfl = std::stod(ary[1]);
    } else if (ary[0]=="c") {
      c = std::stod(ary[1]);
    } else if (ary[0]=="integrator") {
      integrator = ary[1];
    } else if (ary[0]=="icmodel") {
      icmodel = ary[1];
    } else if (ary[0]=="xorder") {
      xorder = ary[1];
    } else if (ary[0]=="ofilename") {
      ofilename = ary[1];
    } else if (ary[0]=="nx1") {
      nx1 = std::stoi(ary[1]);
    } else if (ary[0]=="tlim") {
      tlim = std::stod(ary[1]);
    } else {
      continue;
    }
  }
  ifs.close(); // close the file

  // setting for time integration
  Real time = 0.0;
  
  int nout = 80;
  IntegratorWeight stage_wghts[MAX_NSTAGES];
  int nstages=0;
  if (integrator == "rk2"){
    std::cout << "integrator: " << integrator << std::endl;
    nstages = 2;
    stage_wghts[0].beta    = 1.0; // weight for dt*divF
    stage_wghts[1].beta    = 0.5; // weight for dt*divF    
  } else if (integrator == "LW"){
    std::cout << "integrator: " << integrator << std::endl;
    nstages = 1;
    stage_wghts[0].beta    = 1.0; // weight for dt*divF
  } else if (integrator == "rk3"){
    std::cout << "integrator: " << integrator << std::endl;
    nstages = 3;
    stage_wghts[0].beta    = 1.0; // weight for dt*divF
    stage_wghts[1].beta    = 0.25; // weight for dt*divF
    stage_wghts[2].beta    = 2.0/3.0; // weight for dt*divF
  }

  
  // Coodinates
  Real x1min = -1.0;
  Real x1max = 1.0;

  int ng=0;
  if (xorder=="1"){
    ng = 2;
  } else if ((xorder=="2a") || (xorder=="2b") || (xorder=="2c") || (xorder=="2d")){
    ng = 2;
  } else if (xorder == "3") {
    ng = 2;
  } else if ((xorder == "4a") || (xorder=="5")) {
    ng = 3;
  }
  
  int ncells1 = nx1 + 2*ng;

  Reconstruction *prcn;
  prcn = new Reconstruction(ncells1);
  
  AthenaArray<Real> dx1f, x1f; // face   spacing and positions
  AthenaArray<Real> dx1v, x1v; // volume spacing and positions

  int il = ng;
  int iu = nx1 + ng - 1;

  dx1f.NewAthenaArray(ncells1);
  x1f.NewAthenaArray(ncells1+1);
  
  dx1v.NewAthenaArray(ncells1);
  x1v.NewAthenaArray(ncells1);
  
  Real dx = (x1max - x1min) / (iu - il + 1); // uniform grid
  for (int i=il-ng;i<=iu+ng+1;++i) {
    Real xnorm = static_cast<Real>(i-il) / static_cast<Real>(nx1);
    Real rw = xnorm;
    Real lw = 1.0 - xnorm;
    x1f(i) = x1min * lw + x1max * rw;
  }
  // to make it sure that the boundary locations are correctly set.
  x1f(il) = x1min;
  x1f(iu+1) = x1max;
  std::cout << iu << std::endl;
  
  for (int i=il-ng;i<=iu+ng;++i){
    dx1f(i) = x1f(i+1) - x1f(i);
  }

  for (int i=il-ng; i<=iu+ng; ++i) {
    x1v(i) = 0.5*(x1f(i+1)+x1f(i)); // x1f(i) is defined at i-1/2
  }
  for (int i=il-ng; i<=iu+ng-1; ++i) {
    dx1v(i) = x1v(i+1)-x1v(i);
  }

  
  AthenaArray<Real> u,un,flx,dflx,uan;
  u.NewAthenaArray(ncells1);
  un.NewAthenaArray(ncells1);
  flx.NewAthenaArray(ncells1+1);
  dflx.NewAthenaArray(ncells1);
  uan.NewAthenaArray(ncells1);
  
  // Initial Condition
  SetInitialCondition(u,x1v,x1min,x1max,il,iu,icmodel);

  // Set BC
  SetBoundaryCondition(u,il,iu,ng);
  uan = u; // save initial condition for calculating error (deep copy)

  // First output
  int nwrite = 0;
  int ncycle=0;
  WriteData(nwrite,u,x1v,dx1v,time,ncycle,ncells1);
  
  // Time integration
  Real dt = cfl * dx / std::abs(c);
  std::cout << "dt: " << dt << std::endl;
  Real dt_out = tlim/static_cast<Real>(nout);
  Real time_out = dt_out;

  std::cout << std::fixed << std::setprecision(17) << "Time = " << time << std::endl;
  while(time < tlim){
    un = u; // save u^n (deep copy)
    for (int stage=1;stage<=nstages;++stage){
      // Calculate numerical flux
      CalculateFluxes(flx,u,c,dt,dx,il,iu,xorder,prcn);

      // Calculate weighted-average of u before flux div term is added
      WeightedAve(u,un,il,iu,stage,integrator);

      // Add flux divergence
      Real wght_flux = stage_wghts[stage-1].beta * dt;
      for (int i=il; i<=iu; ++i){
        dflx(i) = (flx(i+1) - flx(i));
      }
      for (int i=il; i<=iu; ++i){
        u(i) -= wght_flux*dflx(i)/dx1f(i);
      }

      // Set BC
      SetBoundaryCondition(u,il,iu,ng);
    }

    time += dt;
    ncycle +=1;
    if (time < tlim && (tlim - time) < dt) {
      dt = tlim - time; // change dt to exactly stop at time = tlim
    }

    // Output
    if(time >= time_out){
      std::cout << std::fixed << std::setprecision(17) << "Time = " << time << std::endl;
      time_out += dt_out;
      WriteData(nwrite,u,x1v,dx1v,time,ncycle,ncells1);
    }
  }
  
  // calculate error
  CalculateError(u,uan,x1v,dx1v,time,il,iu);

  // delete arrays
  dx1f.DeleteAthenaArray();
  x1f.DeleteAthenaArray();
  dx1v.DeleteAthenaArray();
  x1v.DeleteAthenaArray();
  u.DeleteAthenaArray();
  uan.DeleteAthenaArray();

  return 0;
}

void CalculateError(AthenaArray<Real> &u, AthenaArray<Real> &uan, AthenaArray<Real> &x1v,
  AthenaArray<Real> &dx1v, const Real time, const int il, const int iu){
    Real error=0;
    for (int i=il; i<=iu; ++i){
      error += std::abs(u(i)-uan(i));
    }
    error /= (iu-il+1);
    std::cout << "Nmesh, error = " << (iu-il+1) << " " << error << std::endl;
    return;
}

Real func_G(Real x, Real b, Real z){
  Real res = exp(-b*SQR(x-z));
  return res;
}

Real func_F(Real x, Real alp, Real a){
  Real tmp = std::max(1.0-SQR(alp*(x-a)),0.0);
  return sqrt(tmp);
}

void WeightedAve(AthenaArray<Real> &u_out, AthenaArray<Real> &u0, const int is, const int ie,
                 const int s, const std::string integrator){
  if (integrator == "rk2") { // SSPRK(2,2), eq(3.1) of Gottlieb+2009
    if (s==1){
      for (int i=is; i<=ie; ++i){
        u_out(i) = u0(i);
      }
    } else if (s==2) {
      for (int i=is; i<=ie; ++i){
        u_out(i) = 0.5 * u0(i) + 0.5 * u_out(i);
      }
    }
  } else if (integrator == "LW") { // 2-step Lax-Wendroff
    for (int i=is; i<=ie; ++i){
      u_out(i) = u0(i);
    }
  } else if (integrator == "rk3") { // SSPRK(3,3), eq(3.2) of Gottlieb+2009    
    if (s==1){
      for (int i=is; i<=ie; ++i){
        u_out(i) = u0(i);
      }
    } else if (s==2) {
      for (int i=is; i<=ie; ++i){
        u_out(i) = 3.0/4.0 * u0(i) + 0.25 * u_out(i);
      }
    } else if (s==3) {
      for (int i=is; i<=ie; ++i){
        u_out(i) = 1.0/3.0 * u0(i) + 2.0/3.0 * u_out(i);
      }      
    }
  }
  
  
  return;
}

void CalculateFluxes(AthenaArray<Real> &flux, AthenaArray<Real> &u,
                     const Real v, const Real dt, const Real dx, const int is, const int ie, const std::string order,
                     Reconstruction *prcn) {

  AthenaArray<Real> &ul = prcn->ul_, &ur = prcn->ur_;

  // flux(i) is defined at i-1/2
  if (order == "1") {
      for (int i=is; i<=ie+1; ++i) {
        flux(i) = 0.5 * v * (u(i-1)+u(i)) - 0.5 * std::abs(v) * (u(i)-u(i-1));
      }
  } else if (order == "2a") { // Lax-Wendroff
    Real nu = v * dt/dx;
    for (int i=is; i<=ie+1; ++i) {
      flux(i) = v * (u(i-1) + 0.5*(1.0-nu)*(u(i)-u(i-1)) );
    }    
  } else if (order == "2b") { // minmod
    AthenaArray<Real> &uc = prcn->uc_, &dul = prcn->dul_, &dur = prcn->dur_, &dum = prcn->dum_;

    for (int i=is-1; i<=ie+1; ++i) {
      dul(i) = u(i  ) - u(i-1);
      dur(i) = u(i+1) - u(i);
      uc(i)  = u(i);
    }
    for (int i=is-1; i<=ie+1; ++i) {
      Real a = dul(i);
      Real b = dur(i);
      dum(i) = 0.5*( SIGN(a) + SIGN(b) ) *
        std::min( std::abs(a), std::abs(b) );
    }
    for (int i=is-1; i<=ie+1; ++i) {
      ul(i+1) = uc(i) + 0.5*dum(i);
      ur(i)   = uc(i) - 0.5*dum(i);
    }
    for (int i=is; i<=ie+1; ++i) {
      // Upwind (When v is constant, this is equivalent to Rusanov or local Lax-Friedrich)      
      flux(i) = 0.5 * v * (ul(i)+ur(i)) - 0.5*std::abs(v)*(ur(i)-ul(i));
    }
  } else if (order == "2c") { // monotonized central (MC) limiter
    AthenaArray<Real> &uc = prcn->uc_, &dul = prcn->dul_, &dur = prcn->dur_, &dum = prcn->dum_;

    for (int i=is-1; i<=ie+1; ++i) {
      dul(i) = u(i  ) - u(i-1);
      dur(i) = u(i+1) - u(i);
      uc(i)  = u(i);
    }
    for (int i=is-1; i<=ie+1; ++i) {
      Real a = dul(i);
      Real b = dur(i);
      dum(i) = 0.5*( SIGN(a) + SIGN(b) ) *
        std::min( {2.0*std::abs(a), 0.5*std::abs(a+b), 2.0*std::abs(b)} );
    }
    for (int i=is-1; i<=ie+1; ++i) {
      ul(i+1) = uc(i) + 0.5*dum(i);
      ur(i)   = uc(i) - 0.5*dum(i);
    }
    for (int i=is; i<=ie+1; ++i) {
      // Upwind (When v is constant, this is equivalent to Rusanov or local Lax-Friedrich)      
      flux(i) = 0.5 * v * (ul(i)+ur(i)) - 0.5*std::abs(v)*(ur(i)-ul(i));
    }
  } else if (order == "2d") { // superbee
    AthenaArray<Real> &uc = prcn->uc_, &dul = prcn->dul_, &dur = prcn->dur_, &dum = prcn->dum_;

    for (int i=is-1; i<=ie+1; ++i) {
      dul(i) = u(i  ) - u(i-1);
      dur(i) = u(i+1) - u(i);
      uc(i)  = u(i);
    }
    for (int i=is-1; i<=ie+1; ++i) {
      Real a = dul(i);
      Real b = dur(i);
      Real tmp1 = std::min(2.0*std::abs(a), std::abs(b));
      Real tmp2 = std::min(std::abs(a), 2.0*std::abs(b));
      dum(i) = 0.5*( SIGN(a) + SIGN(b) ) * std::max(tmp1,tmp2);
    }
    for (int i=is-1; i<=ie+1; ++i) {
      ul(i+1) = uc(i) + 0.5*dum(i);
      ur(i)   = uc(i) - 0.5*dum(i);
    }
    for (int i=is; i<=ie+1; ++i) {
      // Upwind (When v is constant, this is equivalent to Rusanov or local Lax-Friedrich)      
      flux(i) = 0.5 * v * (ul(i)+ur(i)) - 0.5*std::abs(v)*(ur(i)-ul(i));
    }
  } else if (order == "3") {
    // not prepared yet.
  } else if (order == "4a") { // 4th order version of WENO-Z5, Borges et al. 2008
    AthenaArray<Real> &beta0 = prcn->beta0_, &beta1 = prcn->beta1_, &beta2 = prcn->beta2_,
      &alpha0 = prcn->alpha0_, &alpha1 = prcn->alpha1_, &alpha2 = prcn->alpha2_,
      &wz0 = prcn->wz0_, &wz1 = prcn->wz1_, &wz2 = prcn->wz2_, &tau5 = prcn->tau5_;

    //Real eps = 1e-6;
    Real eps = 1e-40;
    
    //-----------------------------------------------------------------------------
    // for Left upwind interface value at i+1/2    
    //-----------------------------------------------------------------------------
    // left-biased stencil, eq(9)
    for (int i=is-1; i<=ie+1; ++i) {
      Real uxx0 = 0.5 * (u(i-2) - 2.0*u(i-1) + u(i));
      Real ux0  = 0.5 * (u(i-2) - 4.0*u(i-1) + 3.0*u(i));
      beta0(i) = 13.0/3.0 * SQR(uxx0) + SQR(ux0);
    }
    // center biased stencil, eq(10)
    for (int i=is-1; i<=ie+1; ++i) {
      Real uxx1 = 0.5 * (u(i-1) - 2.0*u(i) + u(i+1));
      Real ux1  = 0.5 * (u(i-1) - u(i+1));
      beta1(i) = 13.0/3.0 * SQR(uxx1) + SQR(ux1);
    }
    // right-biased stencil, eq(11)
    for (int i=is-1; i<=ie+1; ++i) {
      Real uxx2 = 0.5 * (u(i) - 2.0*u(i+1) + u(i+2));
      Real ux2  = 0.5 * (u(i+2) - 4.0*u(i+1) + 3.0*u(i));
      beta2(i) = 13.0/3.0 * SQR(uxx2) + SQR(ux2);
    }

    // eq(25)
    for (int i=is-1; i<=ie+1; ++i) {
      tau5(i) = std::abs(beta0(i)-beta2(i));
    }

    // coefficients shown below eq(7)
    // Shu 2009, Jun Luo et al. 2013, etc
    Real d0 = 1.0/10.;
    Real d1 = 3.0/5.0;
    Real d2 = 3.0/10.;

    // Real d0 = 3.0/10.;
    // Real d1 = 3.0/5.0;
    // Real d2 = 1.0/10.;

    // unnormalized weights, alpha. eq(28)
    for (int i=is-1; i<=ie+1; ++i) {
      // alpha0(i) = d0 * pow((1.0 / (beta0(i)+eps)),2.0);
      // alpha1(i) = d1 * pow((1.0 / (beta1(i)+eps)),2.0);
      // alpha2(i) = d2 * pow((1.0 / (beta2(i)+eps)),2.0);
      alpha0(i) = d0 * (1.0 + tau5(i) / (beta0(i)+eps));
      alpha1(i) = d1 * (1.0 + tau5(i) / (beta1(i)+eps));
      alpha2(i) = d2 * (1.0 + tau5(i) / (beta2(i)+eps));
    }

    // eq(28)
    for (int i=is-1; i<=ie+1; ++i) {
      Real denom = 1.0 / (alpha0(i)+alpha1(i)+alpha2(i));
      wz0(i) = alpha0(i) * denom;
      wz1(i) = alpha1(i) * denom;
      wz2(i) = 1.0 - (wz0(i)+wz1(i));
      //wz2(i) = alpha2(i) * denom;
    }

    // WENO reconstruction step, eqs (4) and (5)
    // f_(i+1/2) = sum_k(w_k * f^k_(i+1/2))
    // f^k_(i+1/2) is given by eq(5). See also Jiang & Shu 1996, table 1, for r=3.
    // f^k_(i+1/2) = sum_l(c_(k,l) * f_(i-k+l))
    // c_(k,l) = a_(k,l) in Jiang & Shu 1996.
    // For r=3 (three stencil points),
    // k      l=0     l=1     l=2
    // 0      1/3    -7/6     11/6
    // 1     -1/6     5/6     1/3
    // 2      1/3     5/6    -1/6
    for (int i=is-1; i<=ie; ++i){
      // Shu 2009 WENO review
      // OK?? Xinrong Su et al., Efficient implementation of WENO Scheme on structured meshes
      Real uk0 =  1.0/3.0 * u(i-2) - 7.0/6.0 * u(i-1) + 11.0/6.0 * u(i);
      Real uk1 = -1.0/6.0 * u(i-1) + 5.0/6.0 * u(i)   + 1.0/3.0  * u(i+1);
      Real uk2 =  1.0/3.0 * u(i)   + 5.0/6.0 * u(i+1) - 1.0/6.0  * u(i+2);

      ul(i+1) = wz0(i) * uk0 + wz1(i) * uk1 + wz2(i) * uk2; // NB: ul(i) is defined at i-1/2
    }

    //-----------------------------------------------------------------------------
    // for Right upwind interface value at i+1/2
    //-----------------------------------------------------------------------------

    // right-biased stencil, eq(9)
    for (int i=is-1; i<=ie; ++i) {
      Real uxx0 = 0.5 * (u(i+3) - 2.0*u(i+2) + u(i+1));
      Real ux0  = 0.5 * (u(i+3) - 4.0*u(i+2) + 3.0*u(i+1));
      beta0(i) = 13.0/3.0 * SQR(uxx0) + SQR(ux0);
    }

    // center biased stencil, eq(10)
    for (int i=is-1; i<=ie; ++i) {
      Real uxx1 = 0.5 * (u(i+2) - 2.0*u(i+1) + u(i));
      Real ux1  = 0.5 * (u(i+2) - u(i));
      beta1(i) = 13.0/3.0 * SQR(uxx1) + SQR(ux1);
    }

    // left-biased stencil, eq(11)
    for (int i=is-1; i<=ie; ++i) {
      Real uxx2 = 0.5 * (u(i+1) - 2.0*u(i) + u(i-1));
      Real ux2  = 0.5 * (u(i-1) - 4.0*u(i) + 3.0*u(i+1));
      beta2(i) = 13.0/3.0 * SQR(uxx2) + SQR(ux2);
    }
    
    
    // eq(25)
    for (int i=is-1; i<=ie+1; ++i) {
      tau5(i) = std::abs(beta0(i)-beta2(i));
    }
    
    // coefficients shown below eq(7)
    d0 = 1.0/10.;
    d1 = 3.0/5.0;
    d2 = 3.0/10.;

    // d0 = 3.0/10.;
    // d1 = 3.0/5.0;
    // d2 = 1.0/10.;

    // unnormalized weights, alpha. eq(28)
    for (int i=is-1; i<=ie+1; ++i) {
      // alpha0(i) = d0 * pow((1.0 / (beta0(i)+eps)),2.0);
      // alpha1(i) = d1 * pow((1.0 / (beta1(i)+eps)),2.0);
      // alpha2(i) = d2 * pow((1.0 / (beta2(i)+eps)),2.0);
      alpha0(i) = d0 * (1.0 + tau5(i) / (beta0(i)+eps));
      alpha1(i) = d1 * (1.0 + tau5(i) / (beta1(i)+eps));
      alpha2(i) = d2 * (1.0 + tau5(i) / (beta2(i)+eps));
    }

    // eq(28)
    for (int i=is-1; i<=ie+1; ++i) {
      Real denom = 1.0 / (alpha0(i)+alpha1(i)+alpha2(i));
      wz0(i) = alpha0(i) * denom;
      wz1(i) = alpha1(i) * denom;
      wz2(i) = 1.0 - (wz0(i)+wz1(i));
    }

    // WENO reconstruction step, eqs (4) and (5)
    // f_(i-1/2) = sum_k(w_k * f^k_(i-1/2))
    // f^k_(i-1/2) is given by eq(5).
    // See Jiang & Shu 1996, table 1, for r=3.
    // and a WENO review by Shu 2009
    // "High Order Weighted Essentially Non-Oscillatory Schemes
    //  for Convection Dominated Problems"

    for (int i=is; i<=ie; ++i){
      // I checked that u(x_(i+1/2)) - u^k = O(dx^3)
      Real uk0 = 11.0/6.0 * u(i+1) - 7.0/6.0 * u(i+2) + 1.0/3.0 * u(i+3);
      Real uk1 =  1.0/3.0 * u(i)   + 5.0/6.0 * u(i+1) - 1.0/6.0 * u(i+2);
      Real uk2 = -1.0/6.0 * u(i-1) + 5.0/6.0 * u(i)   + 1.0/3.0 * u(i+1);

      ur(i+1) = wz0(i) * uk0 + wz1(i) * uk1 + wz2(i) * uk2; // NB: ur(i) is defined at i-1/2
    }

    for (int i=is; i<=ie+1; ++i) {
      // Upwind (When v is constant, this is equivalent to Rusanov or local Lax-Friedrich)
      flux(i) = 0.5 * v * (ul(i)+ur(i)) - 0.5*std::abs(v)*(ur(i)-ul(i)); 
    }
  }
  
  return;
}


std::string ZeroPadNumber(int num, int numlen) {
  std::ostringstream ss;
  ss << std::setw(numlen) << std::setfill('0') << num;
  return ss.str();
}

void WriteData(int &nwrite, AthenaArray<Real> &u, AthenaArray<Real> &x, AthenaArray<Real> &dxv,
               const Real t, const int ncycle, const int ncells1){

  std::cout << ofilename << ZeroPadNumber(nwrite,5) << std::endl;
  
  std::string filename = "data/" + ofilename + ZeroPadNumber(nwrite,5)+".dat";
  std::ofstream ofs(filename.c_str());
  if (!ofs){
    std::cerr << "File open failed" << std::endl;
    std::exit(1);
  }

  // output time
  ofs << "# time=";
  ofs << std::fixed << std::setprecision(17); // fix the number of digits to be 17
  ofs << t;
  ofs << "  cycle=" << ncycle;
  ofs << "  variables=u" << std::endl;
  
  ofs << "# i    x    dx    u" << std::endl;
  for (int i=0; i<ncells1; ++i){
    ofs << std::left << std::setw(5) << i << "    "; // left justified
    ofs << std::fixed << std::setprecision(17); // fix the number of digits to be 17
    ofs << x(i) << "    ";
    ofs << dxv(i) << "    ";
    ofs << u(i) << std::endl;
  }

  // ofs << "# "; // comment sign for gnuplot
  // ofs << std::fixed << std::setprecision(17); // fix the number of digits to be 17
  // ofs << t << std::endl;
  
  // for (int i=0; i<ncells1; ++i){
  //   ofs << std::left << std::setw(5) << i << "    "; // left justified
  //   ofs << std::fixed << std::setprecision(17); // fix the number of digits to be 17
  //   ofs << x(i) << "    ";
  //   ofs << dxv(i) << "    ";
  //   ofs << u(i) << std::endl;
  // }


  ofs.close();

  nwrite += 1;
  
  return;
}

void SetBoundaryCondition(AthenaArray<Real> &u, const int il, const int iu, const int ng) {
  // left side
  for (int i=1; i<=ng; ++i){
    u(il-i) = u(iu+1-i);
  }
  
  // right side
  for (int i=1; i<=ng; ++i){
    u(iu+i) = u(il-1+i);
  }
  
  return;
}

void SetInitialCondition(AthenaArray<Real> &u, AthenaArray<Real> &x1v,
                         const Real x1min, const Real x1max,
                         const int il, const int iu, std::string icmodel) {

  if (icmodel=="SineWave") {
    Real amp = 0.1;
    int nmode = 2;
    Real width = x1max-x1min;
    for (int i=il;i<=iu;++i){
      Real x = x1v(i);
      u(i) = amp*sin(2.0*PI*x/width*nmode);
    }
  } else if (icmodel=="ComplexWave") {
    // Initial Condition given in Jiang & Shu 1996
    Real a = 0.5;
    Real z = -0.7;
    Real delta = 0.005;
    Real alpha = 10.;
    Real beta = log(2.)/(36.*SQR(delta));
  
    for (int i=il;i<=iu;++i){
      Real x = x1v(i);
      Real G1 = func_G(x,beta,z-delta);
      Real G2 = func_G(x,beta,z);
      Real G3 = func_G(x,beta,z+delta);
      
      Real F1 = func_F(x,alpha,a-delta);
      Real F2 = func_F(x,alpha,a);
      Real F3 = func_F(x,alpha,a+delta);
      
      Real u0 = 0.0;
      if (x >= -0.8 && x <= -0.6) {
        u0 = 1.0/6.0 * (G1 + G3 + 4.0 * G2);
      } else if (x >= -0.4 && x <= -0.2) {
        u0 = 1.0;
      } else if (x >= 0.0 && x <= 0.2) {
        u0 = 1.0 - std::abs(10.0 * (x-0.1));
      } else if (x >= 0.4 && x <= 0.6) {
        u0 = 1.0/6.0 * (F1 + F3 + 4.0 * F2);
      }
      u(i) = u0;
    }
  }
  
  return;
}
