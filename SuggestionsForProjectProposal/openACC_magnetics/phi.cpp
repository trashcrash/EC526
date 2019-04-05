//================================================================================= 
// This is a 2d phi 4th code on a torus desinged to be easy to convet to openACC.
//=================================================================================

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <chrono>
using namespace std;
#include <cmath>
#include <complex>

#define L 64
#define D 2
 
#define MEASURE 100
#define WARM_UP 1000
typedef complex<double> Complex;
#define I Complex(0,1)
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586


// Forward declarations
//--------------------------------------------------------------------------
typedef struct{
  int Latsize;    //size of lattice
  double dt;      //Leapfrog integrations step size
  int nstep;      //number of HMC steps
  int nskip;      //Trajectories to skip each measurement
  int nperc;      //percolate and flip every nperc
  double lambda;  //phi**4 coupling
  double musqr;   //phi**2 coupling
} param_t;

void printLattice(const double phi[L][L]);
void hotStart(double phi[L][L], param_t p);
void coldStart(double phi[L][L],param_t p);
void writeLattice(const double phi[L][L], fstream & nameStream);
void readLattice(double phi[L][L],  fstream & nameStream );
int hmc(double phi[L][L], param_t p, int iter);
double calcH(double mom[L][L], double phi[L][L],param_t p);
void forceU(double fU[L][L], double phi[L][L], param_t p);
void update_mom(double mom[L][L], double fU[L][L], param_t p, double dt);
void update_phi(double phi[L][L], double mom[L][L], param_t p, double dt);
void trajectory(double mom[L][L], double phi[L][L], param_t p);

void gaussReal_F(double mom[L][L]);
double measMag(const double phi[L][L]);
void LatticePercolate(bool bond[L][L][2*D], int label[L][L], const double phi[L][L]);
bool MultigridSW( int label[L][L],const bool bond[L][L][2*D]);
void FlipSpins(double phi[L][L], const int label[L][L]);

// Global variables for HMC diagnostics
//--------------------------------------------------------------------------
double ave_expmdH = 0.0;
double ave_dH = 0.0;

// BLAS utilities
//--------------------------------------------------------------------------
// Zero lattice field.
template<typename T> inline void zeroField(T phi[L][L]) {
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      phi[x][y] = 0.0;
}

// Copy lattice field
template<typename T> inline void copyField(T phi2[L][L],T phi1[L][L]) {
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      phi2[x][y] = phi1[x][y];
}

// Add Equ v2 += v1 lattice field
template<typename T> inline void addEqField(T phi2[L][L],T phi1[L][L]) {
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      phi2[x][y] += phi1[x][y];
}

// Add Equ square of real b dot b lattice field
template<typename T> inline T sqrSumField(T b[L][L]) {
  T square = (T) 0.0;
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      square += b[x][y]*b[x][y];
  return square;
}

// Add Equ conj(v2) dot v1 lattice field
template<typename T> inline T dotField(T phi1[L][L], T phi2[L][L]) {
  T scalar = (T) 0.0;
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      scalar +=  conj(phi1[x][y])*phi2[x][y];
  return scalar;
}

void hotStart(double phi[L][L],param_t p) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++) {
      phi[x][y] = 2.0*drand48() - 1.0;
      if(drand48() < 0.5) phi[x][y] = - phi[x][y];   
    }
  return;
}  



int main(int argc, char **argv) {
  
  param_t p;
  p.Latsize = L;
  p.nstep = 25;
  p.dt = 1.0/p.nstep;
  p.lambda = 0.125; 
  p.musqr = -0.35;
  p.nskip = 100;
  p.nperc = 5;
  
  double phi[L][L];

  cout <<" Size "<< L  <<" lambda = "<< p.lambda <<" musqr = "<< p.musqr << endl;
  cout<<" time step = "<< p.dt<< " trajectory steps "<<  p.nstep  << " trajectory length = " << p.dt*p.nstep<< endl;
  
  // coldStart(phi,p);
  hotStart(phi, p);
  
  int count = 0;
  int index = 0;
  string namePhiField;
  fstream outPutFile;

  /*
    namePhiField = "Phi_L";
    namePhiField += to_string(p.Latsize) +"_traj"+ to_string(MEASURE*p.nskip + WARM_UP) +"_lambda" + to_string(p.lambda);
    namePhiField += "_musqr" + to_string(p.musqr) + "_dt"+ to_string(p.dt) +"_n"+ to_string(p.nstep) +".dat";
    
    readLattice(initPhase,inPutFile);
    inPutFile.close();  
    cout <<"Test read/write  "<< "\n";
    for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
    cout << initPhase[x][y] <<"\n";
    cout <<"Test read/write  "<< "\n";
    inPutFile.close();
  */
  
  int accepted = 0;
  int measurement = 0;
  double getMag = 0.0;
  double Phi = 0.0;
  double Phi2 = 0.0;
  double Phi4 = 0.0;
  bool stop = false;
  bool bond[L][L][2*D];
  int label[L][L];

  double momZero[L][L];
  zeroField(momZero);
  
  //Thermalise the field. This evolves the phi field
  //from a completely random (hot start)
  //or a completely ordered (cold start)
  //to a state of thermal equlibrium.
  for(int iter = 0; iter < WARM_UP; iter++) {
    accepted = hmc(phi, p, iter);
    if(iter%p.nperc == 0) {
      //LatticePercolate(bond, label, phi);
      stop = false;
      for(int relax = 0; relax < 1000 && !stop; relax++) {
	stop =  MultigridSW(label, bond);
	//FlipSpins(phi,label);
      }
    }  
  }
  
  //reset acceptance
  accepted = 0;
  for(int iter = WARM_UP; iter < p.nskip*MEASURE + WARM_UP; iter++) {
    
    accepted += hmc(phi, p, iter);
    
    //Percolate and flip the lattice
    if(iter%p.nperc ==  0) {
      LatticePercolate(bond, label, phi);
      
      stop = false;
      for(int relax = 0; relax < 1000 && !stop; relax++) {
	stop = MultigridSW(label, bond);
      }

      //Lattice bonds identified. Flip 'em!
      FlipSpins(phi,label);
    }
    
    if((iter+1)%p.nskip == 0) {
      
      measurement++;
      
      //Get observables
      getMag = measMag(phi);
      Phi += getMag;
      
      getMag *= getMag;
      Phi2 += getMag;
      
      getMag *= getMag;
      Phi4 += getMag;      

      double avPhi = Phi/measurement;
      double avPhi2 = Phi2/measurement;
      double avPhi4 = Phi4/measurement;

      double vol = L*L;
      double vol2= vol*vol;
      double vol4= vol*vol*vol*vol;
      
      //Diagnostics and observables
      //---------------------------------------------------------------------
      // 1. Try to keep the acceptance rate between 0.75 - 0.85. Do this by 
      //    varying the step size of the HMC integrator.
      // 2. <exp(-dH)> should be ~1.0, and ever so slightly larger.
      // 3. <dH> should be ~0.0, and ever so slightly positive.
      // 4. <phi>, <phi**2>, <phi**4> and the Binder cumulant depend on
      //    musqr and lambda.
      cout << "measurement " << measurement << endl;
      cout << "HMC acceptance rate = " << (double)accepted/(measurement*p.nskip) << endl;
      cout << "HMC <exp(-dH)>      = " << ave_expmdH/(measurement*p.nskip) << endl;
      cout << "HMC <dH>            = " << ave_dH/(measurement*p.nskip) << endl;
      cout << "MEAS <phi>          = " << setprecision(12) << avPhi/vol << endl;
      cout << "MEAS <phi**2>       = " << setprecision(12) << avPhi2/vol2 << endl;
      cout << "MEAS <phi**4>       = " << setprecision(12) << avPhi4/vol4 << endl;
      cout << "Binder Cumulant     = " << setprecision(12) << 1.0 - Phi4/(3.0*Phi2*Phi2/measurement) << endl;

    }
  }

  //Write lattice to file at end of run.
  outPutFile.open(namePhiField,ios::in|ios::out|ios::trunc);
  outPutFile.setf(ios_base::fixed, ios_base::floatfield);   
  writeLattice(phi, outPutFile);
  outPutFile.close();
  
  return 0;
}

void LatticePercolate(bool bond[L][L][2*D], int label[L][L], const double phi[L][L]) {
  
  double probability; 
  
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      for(int mu  = 0; mu < 2*D;mu++)
	bond[x][y][mu] = false;
  
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++){
      probability = 1.0 -  exp( -2.0 * phi[x][y]*phi[(x + 1)%L][y]);
      if (drand48() < probability) {
	bond[x][y][0] = true;
	bond[(x+1)%L][y][2] = true;
      }
      probability = 1.0 -  exp( -2.0 * phi[x][y]*phi[x][(y+1)%L]);
      if(drand48() < probability) {
	bond[x][y][1] = true;
	bond[x][(y+1)%L][3] = true;
      }
      
      // Random spin on labels = p/m (1, 2, ... L*L):
      // Is it useful to randomize labels?
      
      if(drand48() < 0.5)
	label[x][y] = - (y + x*L + 1);
      else
	label[x][y] =   (y + x*L + 1);
    }
  return;
}

/* 
   Put relax inside the Multigrid
   Single Grid implemenation at present.
   Put in openACC pragma for GPU
*/

bool MultigridSW( int label[L][L],const bool bond[L][L][2*D]) {
  
  bool stop = true;
  int newlabel[L][L];
  int minLabel;
  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      {
	minLabel = label[x][y];  // Find min of connection to local 4 point stencil.
	
	if( bond[x][y][0] && (abs(minLabel) > abs(label[(x+1)%L][y])) )
	  { minLabel = label[(x+1)%L][y];   stop= false;}
	
	if( bond[x][y][1] &&  (abs(minLabel) > abs(label[x][(y+1)%L])) )
	  {  minLabel = label[x][(y +1)%L];  stop= false;}
	
	if( bond[x][y][2] && (abs(minLabel) > abs(label[(x -1 + L)%L][y])) )
	  {  minLabel = label[(x -1 + L)%L][y]; stop= false; }
	
	if( bond[x][y][3] && (abs(minLabel) > abs(label[x][(y -1 + L)%L])) )
	  {   minLabel = label[x][(y -1 + L)%L]; stop= false; }
	
	newlabel[x][y] =  minLabel;
      }
  
  for(int x = 0; x< L; x++)
    for(int y = 0; y< L; y++)
      label[x][y]  =  newlabel[x][y];
  
  return stop;
}

/* 
   Also put FlipSins on GPU. 
*/

void FlipSpins(double phi[L][L], const int label[L][L]) {
  
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++) {
      if(label[x][y] < 0) phi[x][y] = - phi[x][y];
    }  
  return;
}

int hmc(double phi[L][L], param_t p, int iter) {
  
  double mom[L][L];
  double phiOld[L][L];
  double H = 0.0, Hold = 0.0;

  //Copy the field in case of rejection
  copyField(phiOld, phi);
  
  zeroField(mom);  
  gaussReal_F(mom); 
  
  Hold = calcH(mom, phi, p);
  trajectory(mom, phi, p); // MD trajectory using Verlet  
  H = calcH(mom, phi, p);

  //record HMC diagnostics
  if(iter >= WARM_UP) {
    ave_expmdH += exp(-(H-Hold));
    ave_dH += H-Hold;
  }
  
  // Metropolis accept/reject step
  // Always accepts trajectories during first half of warm up.
  if (drand48() > exp(-(H-Hold)) && iter > WARM_UP/2-1) {
    
    //Keep old field
    copyField(phi, phiOld);
    return 0;
  }
  else {
    //Continue with new field
    return 1;
  }
}
 
void gaussReal_F(double field[L][L]) {
  //normalized gaussian exp[ - phi*phi/2]  <eta^2> = 1
  double r, theta;
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++){
      r = sqrt( -2.0*log(drand48()) );
      theta = TWO_PI*drand48();
      field[x][y] = r*cos(theta);
    }
  return;
}

double calcH(double mom[L][L], double phi[L][L],  param_t p) {
  
  double Hphi = 0.0, Hmom = 0.0;  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++) {
      
      Hphi += 0.5*(phi[x][y] - phi[(x+1)%L][y])*(phi[x][y] - phi[(x+1)%L][y]);
      Hphi += 0.5*(phi[x][y] - phi[x][(y+1)%L])*(phi[x][y] - phi[x][(y+1)%L]);
      Hphi += (p.musqr + p.lambda*phi[x][y]*phi[x][y])*phi[x][y]*phi[x][y];
      
      Hmom += 0.5 * mom[x][y] * mom[x][y];
    }
  
  return Hphi + Hmom;
}

/*====================================================
  dphi/dt = mom = dH/dmom    dmom/dt = F = - dH/phi
  Put in openACC pragma   
  =====================================================*/

void trajectory(double mom[L][L], double phi[L][L], param_t p) {

  const double dt = p.dt;
  const int nstep = p.nstep;
  
  double fU[L][L];
  
  //Initial half step:
  //P_{1/2} = P_0 - dtau/2 * fU
  forceU(fU, phi, p);
  update_mom(mom, fU, p, 0.5*dt);
  
  //step loop
  for(int k=1; k<p.nstep; k++) {
    
    //U_{k} = U_{k-1} + P_{k-1/2} * dt
    update_phi(phi, mom, p, dt);
    
    //P_{k+1/2} = P_{k-1/2} - fU * dt 
    forceU(fU, phi, p);
    update_mom(mom, fU,  p, dt);
    
  } //end step loop
  
  //Final half step.
  //U_{n} = U_{n-1} + P_{n-1/2} * dt
  update_phi(phi, mom, p, dt);
  forceU(fU, phi, p);
  update_mom(mom, fU, p, 0.5*dt);
  return;
}

void forceU(double fU[L][L], double phi[L][L], param_t p) {
  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++) {
      fU[x][y] = 0.0;
      fU[x][y] -= phi[(x+1)%L][y] - 2.0*phi[x][y] + phi[(x-1+L)%L][y];
      fU[x][y] -= phi[x][(y+1)%L] - 2.0*phi[x][y] + phi[x][(y-1+L)%L];
      fU[x][y] += (2.0*p.musqr + 4.0*p.lambda*phi[x][y]*phi[x][y]) * phi[x][y];
    }
  return;
}

void update_mom(double mom[L][L], double fU[L][L], param_t p, double dt) {
  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++) 
      mom[x][y] -= fU[x][y] * dt;
}


void update_phi(double phi[L][L], double mom[L][L], param_t p, double dt) {
  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++) 
      phi[x][y] += mom[x][y] * dt;
  
  return;
}




void printLattice(const double phi[L][L]) {
  for(int x =0;x< L;x++){
    for(int y =0;y< L;y++)
      cout <<"("<<x<<","<<y<<") = " << phi[x][y] << endl;
    cout << "\n\n";
  }
  return;
}

double measMag(const double phi[L][L]) {
  double mag = 0.0;
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      mag += phi[x][y];  
  return mag;
}

void coldStart(double phi[L][L],param_t p) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      phi[x][y] = 1.0;
  return;
}

void writeLattice(const double phi[L][L],fstream & outPutFile) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      outPutFile << setprecision(12) <<  setw(20) <<  phi[x][y] <<"\n";
  return;
}

void readLattice(double phi[L][L],fstream & inPutFile ) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++) {
      inPutFile >> phi[x][y];
      cout << phi[x][y] << "\n";
    }
  return;
}
