/* 
================================================================================= 
This is a 2d phi 4th code on a torus desinged to be easy to convet to openACC.
=================================================================================

Comparison of Binder Cummulant with Schaich 

Standard Form:  1/2\sum_<xy> (phi_x - phi_y)^2 +  m_in/2 phi^2_x + lambda_in/4 phi^4_x
This Code:      S = 1/2\sum_<xy> (phi_x - phi_y)^2  - musqr phi^2_x + lambda phi^4_x
Schaich Code:   S = - \sum_<x y> phi_x phi_y  + (2 + m_in/2) phi^2_x  + lambda phi^4_x

So Schaich does a redefinition so simplify the code

lambda = lambda_in / 4;
muSquared = 2 + m_in / 2 ;

So For this code: ./2d_phi4 - 0.7 0.5 32 16384 16384
in the present code the parameters map is

lambda = 0.5 ==> lambda 0.5/4 = 0.125
m_in = -0.7  ==> musqr =  - m_in / 2=  0.7/2   = 0.35

Schaich Binder  = 0.524037

My test value:  = 0.52032117573  and 0.521459040377

With SW  = 0.520920506171

*/


#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;
#include <math.h>     //Std c++ math lib
#include <accelmath.h>//PGI math lib 
#include <complex>  
#include "ran2s.h"

// Change later to Lx and Ly (time on cylinder)
#define L 512
#define D 2
#define MEAS 10
#define HMC_TRAJS 1
#define CLUSTER_SWEEPS 2
#define RELAX 500
#define RELAX_MAX 20000
#define RELAX_INTERVAL 100

//If defined, perform the serial timing comparison
//#define TIMING_COMP

#define MAX_ITER   100
#define WARM_UP    100

typedef complex<double> Complex;
#define I Complex(0,1)
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
static long iseed=347023L;

typedef struct{
  int Latsize;
  int nstep;
  double dt;
  double lambda;
  double musqr;
} param_t;

//Utilities
void printLattice(const double phi[L][L]);
void hotStart(double phi[L][L], param_t p);
void coldStart(double phi[L][L],param_t p);
void systemRands(double rands[L][L], int block);
void HMCsystemRands(double *rands, int length);
void writeLattice(const double phi[L][L], fstream & nameStream);
void readLattice(double phi[L][L],  fstream & nameStream );

//PGI accelerated functions
void twoPtMomSpace(const int cluster_num, const double phiCOPY[L][L],
		   const int label[L][L], double twoPtKmode[L][L]);

int hmc(double phi[L][L], double phiCOPY[L][L], double mom[L][L],
	double rands[L][L], param_t p, int iter);
double calcH(double mom[L][L], double phi[L][L], param_t p);
void trajectory(double mom[L][L], double phi[L][L], param_t p);
void gaussReal_F(double mom[L][L], double rands[L][L], int block);
void gaussReal_F(double mom[L][L], double rands[L][L]);
void FlipSpins(double phi[L][L], const int label[L][L]);
bool MultigridSWTest(int label[L][L], const bool bond[L][L][2*D]);
void MultigridSW(int label[L][L], const bool bond[L][L][2*D], int a);
void LatticePercolate(bool bond[L][L][2*D], int label[L][L], const double phi[L][L],
		      const double rands1[L][L],
		      const double rands2[L][L],
		      const double rands3[L][L]);
double measMag(const double phi[L][L]);

//Serial reference functions
void twoPtMomSpace_serial(const int cluster_num, const double phiCOPY[L][L],
			  const int label[L][L], double twoPtKmode[L][L]);

double calcH_serial(double mom[L][L], double phi[L][L],param_t p);
void trajectory_serial(double mom[L][L], double phi[L][L], param_t p);
void LatticePercolate_serial(bool bond[L][L][2*D], int label[L][L], const double phi[L][L]);
void FlipSpins_serial(double phi[L][L], const int label[L][L]);
bool MultigridSW_serial( int label[L][L],const bool bond[L][L][2*D]);
void gaussReal_F_serial(double mom[L][L], double rands[L][L]);

void forceU(double fU[L][L], const double phi[L][L], param_t p);
void update_mom(double mom[L][L], double fU[L][L], param_t p, double dt);
void update_phi(double phi[L][L], double mom[L][L], param_t p, double dt);
void TestForce(double phi[L][L],param_t p);
double measMag_serial(const double phi[L][L]);

// Utilities and BLAS
//=============================================================================

inline double get_time() {
  
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return 1.0*tv.tv_sec+1.0E-6*tv.tv_usec;

}

// Zero lattice field (Device)
template<typename T> inline void zeroField(T phi[L][L]) {
#pragma acc data present(phi[0:L][0:L])
  {
#pragma acc parallel loop collapse(2)    
    for(int x=0;x<L; x++)
      for(int y=0;y<L; y++)
	phi[x][y] = 0.0;
  }
}

// Zero lattice field (Host)
template<typename T> inline void zeroField_host(T phi[L][L]) {  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      phi[x][y] = 0.0;
}

  
// Copy lattice field (Device)
template<typename T> inline void copyField(T phi2[L][L], T phi1[L][L]) {
  
#pragma acc data present(phi2[0:L][0:L]) present(phi1[0:L][0:L])
  {
#pragma acc parallel loop collapse(2)    
    for(int x=0; x<L;x++)
      for(int y=0; y<L;y++)
	phi2[x][y] = phi1[x][y];
  }
}

// Copy lattice field (Host)
template<typename T> inline void copyField_host(T phi2[L][L], T phi1[L][L]) {

  for(int x=0; x<L;x++)
    for(int y=0; y<L;y++)
      phi2[x][y] = phi1[x][y];
}


// Global diff lattice field
// We indicate that the data is present on the device and unroll the
// nested loop.
template<typename T> inline T diffField(T phi2[L][L], T phi1[L][L]) {
#pragma acc data present(phi2[0:L][0:L]) present(phi1[0:L][0:L])
  {
    T diff = 0.0;
#pragma acc parallel loop reduction(+:diff) collapse(2)
    for(int x=0; x<L;x++)
      for(int y=0; y<L;y++)
	diff += (sqrt(pow(phi2[x][y],2)) - sqrt(pow(phi1[x][y],2)));
    
    return diff;
  }
}

// Global diff lattice field
// We indicate that the data is present on the device and unroll the
// nested loop.
template<typename T> inline T diffField_host(T phi2[L][L], T phi1[L][L]) {
  T diff = 0.0;
  for(int x=0; x<L;x++)
    for(int y=0; y<L;y++) {
      diff += (sqrt(pow(phi2[x][y],2)) - sqrt(pow(phi1[x][y],2)));
      //cout << "(" << x << "," << y << ") " << phi2[x][y] << " " << phi1[x][y] << endl;
    }
  return diff;
}


// Add Equ v2 += v1 lattice field
template<typename T> inline void addEqField(T phi2[L][L],T phi1[L][L]) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
       phi2[x][y] +=  phi1[x][y];
}

// Add Equ square of real b dot b lattice field
template<typename T> inline T sqrSumField(T b[L][L]) {
  T square = (T) 0.0;
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      square +=  b[x][y]*b[x][y];
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

// Dump lattice to stdout
void printLattice(const double phi[L][L]) {
  for(int x=0; x<L; x++){
    for(int y=0; y<L; y++)
      cout << "("<<x<<","<<y<<") = "<<phi[x][y]<<"\n";
    cout << "\n\n";
  }
  return;
}

// Measure the average phi field value
double measMag(const double phi[L][L]) {
  
  double magOut = 0.0;
#pragma acc data present(phi[0:L][0:L])
  {
    double mag = 0.0;
#pragma acc parallel loop reduction(+:mag) collapse(2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++)
	mag += phi[x][y];
    
    magOut = mag;
  }
  return magOut;
}

// Populate lattice with random numbers
void hotStart(double phi[L][L],param_t p) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++) {
      phi[x][y] = 2.0*drand48() - 1.0;
      if(drand48() < 0.5)  phi[x][y] = - phi[x][y];   
    }
  return;
}  

// Populate lattice with unit
void coldStart(double phi[L][L],param_t p) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      phi[x][y] = 1.0;
  return;
}

// Dump lattice to disk
void writeLattice(const double phi[L][L],fstream & outPutFile) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      outPutFile << setprecision(12) <<  setw(20) <<  phi[x][y] <<"\n";
  return;
}

// Read lattice from disk
void readLattice(double phi[L][L], fstream &inPutFile) {
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++) {
      inPutFile >> phi[x][y];
      //cout << phi[x][y] << "\n";
    }
  return;
}
//=============================================================================


//Global variables
//=========================
double HMC_PGI_time = 0.0;
double PGI_total_time = 0.0;
double PGI_expmdHave = 0.0;

#ifdef TIMING_COMP
double HMC_SERIAL_time = 0.0;
double SERIAL_total_time = 0.0;
double SERIAL_expmdHave = 0.0;
#endif
//=========================


// Begin Main Program
//==============================================================
int main(void) {

  //Declare physical parameters
  param_t p;
  p.Latsize = L;
  p.nstep = 1000;
  p.dt = 0.001;
  p.lambda = 0.125; 
  p.musqr = 0.3605;
  sran2(iseed);
  
  cout <<" Size "<< L  <<" lambda = "<< p.lambda <<" musqr = "<< p.musqr <<"  MAX_ITER  =  " << MAX_ITER << "\n";
  cout<<" time step = "<< p.dt<< " trajectory steps "<<  p.nstep  << " traj length = " << p.dt*p.nstep<< endl;

  string namePhiField;
  fstream outPutFile;
  fstream inPutFile;
  
  namePhiField = "Phi_L";
  namePhiField +=to_string(p.Latsize) +"_I"+ to_string(MAX_ITER) +"_lambda" + to_string(p.lambda);
  namePhiField +="_musqr" + to_string(p.musqr) + "_dt"+ to_string(p.dt) +"_n"+ to_string(p.nstep) +".dat";

  //Observables
  double getMag = 0.0;
  double avPhi = 0.0;
  double avPhi2 = 0.0;
  double avPhi4 = 0.0;
  
  //PGI Timing 
  double start = 0.0;
  double PGI_time = 0.0;
  double PERC_PGI_time = 0.0;
  double MGSW_PGI_time = 0.0;
  double TWOPTC_PGI_time = 0.0;
#ifdef TIMING_COMP
  //Serial timing
  double SERIAL_time = 0.0;
  double PERC_SERIAL_time = 0.0;
  double MGSW_SERIAL_time = 0.0;
  double TWOPTC_SERIAL_time = 0.0;
  double twoPtKmode_serial[L][L];
#endif
  
  //Simple FT for 2pt function
  double twoPtKmode[L][L];
  
  int accepted = 0;
  int measurement = 0;
  
  //Arrays to store data. Device arrays are delcared at the start
  //of OPENACC regions
  double phi[L][L];
  double phiCOPY[L][L];  
  bool bond[L][L][2*D];
  int label[L][L];
  double mom[L][L];
  double rands1[L][L];
  double rands2[L][L];
  double rands3[L][L];
#ifdef TIMING_COMP
  //Arrays to hold data specifically for timing and cross checking
  double phi_serial[L][L];
  bool bond_serial[L][L][2*D];
  int label_serial[L][L];
#endif  
  
  //Read lattice
  //inPutFile.open(namePhiField,ios::in|ios::out|ios::trunc);
  //inPutFile.setf(ios_base::fixed,ios_base::floatfield); 
  //readLattice(phi, inPutFile);
  //inPutFile.close();  

  //Random lattice 
  // coldStart(phi,p);
  hotStart(phi, p);
  
  // OpenACC Init 
#pragma acc init
  
  //ENTER ACC Parallel Region
  //======================================================================
  // copyout will create the array on the device and copy host data to
  // the device
  // create will create space on the device only
#pragma acc data copyout(phi[0:L][0:L]) create(bond[0:L][0:L][0:2*D], phiCOPY[0:L][0:L], mom[0:L][0:L], label[0:L][0:L], twoPtKmode[0:L][0:L], rands1[0:L][0:L], rands2[0:L][0:L], rands3[0:L][0:L])
  { 
    
    //Loop over warmup iterations
    for(int iter=0; iter < WARM_UP; iter++) {
      
      //Perform 'HMCtraj' number of HMC sweeps.
      hmc(phi, phiCOPY, mom, rands1, p, iter);

      //Perform 'CLUSTER_SWEEPS' number of cluster updates
      for(int clusters=0; clusters<CLUSTER_SWEEPS; clusters++ ) {
	
	//Compute new random numbers of the host for lattice percolation
	int numblocks = L;
	for(int block=0; block<numblocks; block++) {
	  systemRands(rands1, block);
	  systemRands(rands2, block);
	  systemRands(rands3, block);
	}
	
	//This pragma indicates that we wish to update the arrays on the device
	//with the data we just placed in the corresponding host arrays.
#pragma acc update device(rands1[0:L][0:L], rands2[0:L][0:L], rands3[0:L][0:L])      
	
	//Perform the lattice percolation of the bonds on the device
	LatticePercolate(bond, label, phi, rands1, rands2, rands3);
	bool stop = false;
	
	//Perform the SW cluster construction
	//This first function perform RELAX number of sweeps
	//in parallel. This is done so that individual sweeps do not
	//pause to return a boolean test value to the host
	MultigridSW(label, bond, RELAX);
	for(int relax = RELAX; !stop; relax += RELAX_INTERVAL) {
	  MultigridSW(label, bond, RELAX_INTERVAL-1);

	  //We now switch to a different function that tests to see
	  //if any new bonds are added to the clusters.
	  stop = MultigridSWTest(label, bond);

	  //If stop is returned true, the clusters have been identified.
	  if(stop) {	    
	    FlipSpins(phi, label);
	    //printf("MGSW step %d complete at relax step %d\n", clusters, relax);
	  }

	  //If there are too few relaxation sweeps, the clusters are not
	  //yet properly formed.
	  if(relax > RELAX_MAX) {
	    printf("Error in MGSW. Please increase the number of relaxation sweeps.\n");
	    exit(0);
	  }
	}      
      }
    }

    //Warm up complete. Begin measurement iterations
    //===============================================================================

    zeroField(twoPtKmode);
#ifdef TIMING_COMP
    zeroField_host(twoPtKmode_serial);
#endif
    
    for(int iter = WARM_UP; iter < MAX_ITER + WARM_UP; iter++) {

      //We now measure the acceptance rate of the HMC, and
      //perfrom the serial HMC is timing is defined.
      accepted += hmc(phi, phiCOPY, mom, rands1, p, iter);
      
      for(int clusters=0; clusters<CLUSTER_SWEEPS; clusters++ ) {
	
	start = get_time();
	//Compute new random numbers on the host for lattice percolation
	int numblocks = L;
	for(int block=0; block<numblocks; block++) {
	  systemRands(rands1, block);
	  systemRands(rands2, block);
	  systemRands(rands3, block);
	}
	
	//Update the rands arrays for the percolation routine
#pragma acc update device(rands1[0:L][0:L], rands2[0:L][0:L], rands3[0:L][0:L])

	//Device side lattice percolation
	LatticePercolate(bond, label, phi, rands1, rands2, rands3);
	PGI_time = get_time() - start;
	PERC_PGI_time += PGI_time;
	PGI_total_time += PGI_time;
	
#ifdef TIMING_COMP
	//SERIAL Percolation
	//First, we must update the phi array on the host and copy to
	//the serial phi array. If the HMC checks pass, we know the phi
	//field should be identical anyway.
#pragma acc update self(phi[0:L][0:L])
	copyField_host(phi_serial, phi);
	
	start = get_time();
	LatticePercolate_serial(bond_serial, label_serial, phi_serial);
	SERIAL_time = get_time() - start;
	PERC_SERIAL_time += SERIAL_time;
	SERIAL_total_time += SERIAL_time;
#endif
	
	//PGI MultigridSW and Spin Flip
	start = get_time();

	//Perform the device side SW cluster construction
	bool stop = false;
	MultigridSW(label, bond, RELAX);
	for(int relax = RELAX; !stop; relax += RELAX_INTERVAL) {
	  MultigridSW(label, bond, RELAX_INTERVAL-1);
	  stop = MultigridSWTest(label, bond);
	  if(stop) {
	    FlipSpins(phi, label);
	    //printf("MGSW step %d complete at relax step %d\n", clusters, relax);
	  }
	  if(relax > RELAX_MAX) {
	    printf("Error in MGSW. Please increase the number of relaxation sweeps.\n");
	    exit(0);
	  }
	}      
	
	PGI_time = get_time() - start;
	MGSW_PGI_time += PGI_time;
	PGI_total_time += PGI_time;
	
#ifdef TIMING_COMP
	//Serial MultigridSW and Spin Flip
	start = get_time();
	
	stop = false;
	for(int relax = 0; !stop; relax++) {
	  stop = MultigridSW_serial(label_serial, bond_serial);
	  if(stop) {
	    FlipSpins_serial(phi_serial, label_serial);
	    //printf("MGSW serial complete at relax step %d\n", relax);
	  }
	  if(relax > 10*RELAX) {
	    printf("Error in MGSW. Please increase the number of relaxation sweeps.\n");
	    exit(0);
	  }
	}
	SERIAL_time = get_time() - start;
	MGSW_SERIAL_time += SERIAL_time;
	SERIAL_total_time += SERIAL_time;
	if((iter+1)%MEAS == 0 && clusters == CLUSTER_SWEEPS-1) {
	  printf("TIMING PERCOLATE: PGI time %.6e, serial time %.6e, speedup: %.4f\n", PERC_PGI_time, PERC_SERIAL_time, PERC_SERIAL_time/PERC_PGI_time);
	  printf("TIMING MGSW+FLIP: PGI time %.6e, serial time %.6e, speedup: %.4f\n",
		 MGSW_PGI_time, MGSW_SERIAL_time, MGSW_SERIAL_time/MGSW_PGI_time);
	}

#else
	if((iter+1)%MEAS == 0 && clusters == CLUSTER_SWEEPS-1) {
	  printf("TIMING PERCOLATE: PGI time %.6e\n", PERC_PGI_time);
	  printf("TIMING MGSW+FLIP: PGI time %.6e\n", MGSW_PGI_time);	  
	}
#endif
      }

      
      if((iter+1)%MEAS == 0) {
	
	//Measurement routines
	//------------------------------------------------
	measurement++;

	//Update the phi and label array with data from the device
#pragma acc update self(phi[0:L][0:L], label[0:L][0:L])
	
	//Object to hold the cluster info and the field info.
	//It is a vector of a pair of data types,
	//in this case 'int' and 'double'. The int will hold the
	//cluster value, and phi will hold the field value.
	std::vector<pair<int,double>> label_phi;
	
	//Populate the vector
	for(int x=0; x<L; x++){
	  for(int y=0; y<L; y++) {	    
	    label_phi.push_back( make_pair(abs(label[x][y]), phi[x][y]) );	  
	  }
	}

	//Sort the label-phi pair by label (first in the pair)
	sort(label_phi.begin(), label_phi.end());      
	
	//Rename the clusters with cardinal (1,2,3,...) labels
	//instead of their arbitrary lowest site labels
	int last_cluster = 0;
	int cluster_num = 0;
	int coor = 0;
	for(int x=0; x<L; x++){
	  for(int y=0; y<L; y++) {

	    if(label_phi[coor].first != last_cluster){ 
	      //New cluster encounterd	    
	      cluster_num++;
	      //Reset last_cluster value
	      last_cluster = label_phi[coor].first;
	    }	  
	    //Make this label the current cardinal label
	    label_phi[coor].first = cluster_num;
	    
	    coor++;
	  }
	}

	//We must transfer the (ordered) data from the label_phi vector to a more
	//stable, PGI friendly data array. We may use the phiCOPY array for this
	//purpose, provided we copy back the original phi array to phiCOPY before
	//the next HMC. We must also create and copy the ordered cluster label data
	//to the device. We may use the label array as it will be overwritten anyway
	//at the next SW step.

	coor = 0;
	for(int x=0; x<L; x++){
	  for(int y=0; y<L; y++) {
	    label[x][y]   = label_phi[coor].first;
	    phiCOPY[x][y] = label_phi[coor].second;
	    coor++;
	  }
	}

	//Copy phi and label data to the device
#pragma acc update device(phiCOPY[0:L][0:L], label[0:L][0:L])

	start = get_time();
	twoPtMomSpace(cluster_num, phiCOPY, label, twoPtKmode);
	PGI_time = get_time() - start;
	
	TWOPTC_PGI_time += PGI_time;
	PGI_total_time += PGI_time;
	
#pragma acc update self(twoPtKmode[0:L][0:L])
	
#ifdef TIMING_COMP

	//Perform a dummy computation using a serial routine and the PGI data.
	start = get_time();	
	twoPtMomSpace_serial(cluster_num, phiCOPY, label, twoPtKmode_serial);
	
	SERIAL_time = get_time() - start;
	TWOPTC_SERIAL_time += SERIAL_time;
        SERIAL_total_time += SERIAL_time;	
        printf("TIMING  2PT CORR: PGI time %.6e, serial time %.6e, speedup: %.4f\n",
	       TWOPTC_PGI_time, TWOPTC_SERIAL_time, TWOPTC_SERIAL_time/TWOPTC_PGI_time);
	printf("TIMING CUMULATED: PGI time %.6e, serial time %.6e, speedup: %.4f\n\n",
	       PGI_total_time, SERIAL_total_time, SERIAL_total_time/PGI_total_time);

	
	//Inspect the PGI vs Serial correlation functions
	for(int x=0; x<L; x++){
	  for(int y=0; y<L; y++) {
	    //if(x == y && x < L) printf("%d %d %f %f\n", x, y, twoPtKmode[x][y]/measurement, twoPtKmode_serial[x][y]/measurement);
	  }
	}
#else
	printf("TIMING  2PT CORR: PGI time %.6e\n", TWOPTC_PGI_time);
	printf("TIMING CUMULATED: PGI time %.6e\n\n", PGI_total_time);	
#endif
	
	//Moments of phi analysis
	//PGI routine
	getMag  = measMag(phi);	
	avPhi  += getMag;
	
	getMag *= getMag;
	avPhi2 += getMag;
	
	getMag *= getMag;
	avPhi4 += getMag;
	
	double tphi  = avPhi/measurement;
	double tphi2 = avPhi2/measurement;
	double tphi4 = avPhi4/measurement;
	
	cout<<"HMC rate        " << (1.0*accepted)/((iter+1 - WARM_UP)*HMC_TRAJS) << endl;
	int vol = L*L;
	cout<<"Average M       "<< setprecision(12) << tphi/vol << endl;
	cout<<"Average M**2    "<< setprecision(12) << tphi2/pow(vol,2) << endl;
	cout<<"Average M**4    "<< setprecision(12) << tphi4/pow(vol,4) << endl;
	cout<<"Binder Cumulant "<< setprecision(12) << 1.0 - tphi4/(3.0*tphi2*tphi2) << endl;
	printf("----------------------------------------------------------------\n\n");
	
	//Dump lattice
	//outPutFile.open(namePhiField,ios::in|ios::out|ios::trunc);
	//outPutFile.setf(ios_base::fixed,ios_base::floatfield); 
	//writeLattice(phi,outPutFile);
	//outPutFile.close();      
	
      }
    }
  }//EXIT ACC Parallel Region
    
  return 0;
}


// PGI accelerated routines
//==================================================================================

//Pure gauge Hybrid Monte Carlo
int hmc(double phi[L][L], double phiCOPY[L][L], double mom[L][L],
	double rands[L][L], param_t p, int iter) {
   
  double HPGIold = 0.0;
  double HPGInew = 0.0;
#ifdef TIMING_COMP

  //To keep argument passing simple, we declare local
  //serial arrays here and compare results for each HMC
  //iteration. If the HMC data matches, we may copy the
  //device phi array to the phi_serial array with
  //impunity.
  double HSERIALold = 0.0;
  double HSERIALnew = 0.0;
  double momSERIAL[L][L];
  double phiSERIAL[L][L];
  double SERIAL_time = 0.0;
#endif
  
  int accepted = 0;  
  double PGI_time = 0.0;
  double start = 0.0;
  
  //All arrays are alreay present on the device. The phi arrays are populated, we need
  //only to refresh the random numbers and recreate momenta.
  //BEGIN acc region
#pragma acc data present(phi[0:L][0:L], phiCOPY[0:L][0:L], mom[0:L][0:L], rands[0:L][0:L])
  { 
    
    //Copy of the phi field in case of Metropolis rejection. We also copy the phi
    //field to the host for cross checking
    copyField(phiCOPY, phi);    
#ifdef TIMING_COMP
#pragma acc update self(phi[0:L][0:L])
    copyField_host(phiSERIAL, phi);
#endif
    
    //Loop over trajectories
    for(int i=0; i<HMC_TRAJS; i++) {

      //Do a PGI accelerated trajectory
      start = get_time();
      
      //Populate rands array
      HMCsystemRands((double*)rands, L*L);      
      //Copy random numbers to the device
#pragma acc update device(rands[0:L][0:L])
      //Generate gaussian distributed momemta on the device
      gaussReal_F(mom, rands);                  
      //DMH FIXME. Figure out how to do data transfer to the device while the
      //kernels are executing. The CPU calls to generate random numbers and the
      //GPU function to compute the gaussian momenta can be done concurrently.
      //Is it worth it for the time it saves?

      
      // MD trajectory using Verlet on the device
      HPGIold = calcH(mom, phi, p);
      trajectory(mom, phi, p);      
      HPGInew = calcH(mom, phi, p);

      if(iter >= WARM_UP) PGI_expmdHave += exp(HPGIold - HPGInew);	
      
      //If thermalised, collect timing.
      if(iter >= WARM_UP) {
	PGI_time = get_time() - start;
	HMC_PGI_time += PGI_time;
	PGI_total_time += PGI_time;
      }

      //Do a dummy serial trajectory for timing and cross check
#ifdef TIMING_COMP
      
      if(iter >= WARM_UP) {

	start = get_time();

	//Use the same random numbers to generate the same inital momenta 
	gaussReal_F_serial(momSERIAL, rands);

	HSERIALold = calcH_serial(momSERIAL, phiSERIAL, p);
	trajectory_serial(momSERIAL, phiSERIAL, p); 
	HSERIALnew = calcH_serial(momSERIAL, phiSERIAL, p);	
	SERIAL_expmdHave += exp(HSERIALold - HSERIALnew);

	SERIAL_time = get_time() - start;
	HMC_SERIAL_time += SERIAL_time;
	SERIAL_total_time += SERIAL_time;
      }

      //Dump info to stdout 
      if((iter)%MEAS == 0 && i==0 && iter >= WARM_UP) {
	printf("iter %d\n", iter);
	printf("----------------------------------------------------------------\n");
	printf("   [PGI] exp(-dH) = %.01f - %.01f = %+.12f <exp(-dH)> = %+.06f\n",
	       HPGIold, HPGInew, exp(HPGIold - HPGInew),
	       PGI_expmdHave/(HMC_TRAJS*(iter - WARM_UP) + i + 1));
	if(iter >= WARM_UP) {
	  printf("[SERIAL] exp(-dH) = %.01f - %.01f = %+.12f <exp(-dH)> = %+.06f\n",
		 HSERIALold, HSERIALnew, exp(HSERIALold - HSERIALnew),
		 SERIAL_expmdHave/(HMC_TRAJS*(iter - WARM_UP) + i + 1));
	}
      }
      else if((iter)%MEAS == 0 && i==0 && iter < WARM_UP) {
	printf("iter %d\n", iter);
	printf("----------------------------------------------------------------\n");
	printf("   [PGI] exp(-dH) = %.01f - %.01f = %+.06f\n\n",
	       HPGIold, HPGInew, exp(HPGIold - HPGInew));
      }
      
#else
      if((iter)%MEAS == 0 && i==0 && iter > 0) {
	printf("iter %d:", iter);	
	printf("[PGI] exp(-dH) = %.01f - %.01f = %+.06f <exp(-dH)> = %+.06f\n",
	       HPGIold, HPGInew, exp(HPGIold - HPGInew),
	       PGI_expmdHave/(HMC_TRAJS*(iter - WARM_UP) + i + 1));
      }
#endif
      
      // Check reversibility
      //checkRev(mom, phi, Hold, p);     

      // Metropolis accept/reject step. This must be done on the host so is common
      // to both serial and PGI routines.
      start = get_time();
      if( drand48() > exp(-(HPGInew - HPGIold)) && iter > WARM_UP) {
	//Retain the old lattice
	copyField(phi, phiCOPY);
      }
      else {
	//Update the phi copy
	accepted += 1;
	copyField(phiCOPY, phi);
      }

      //If thermalised, collect timing.
      if(iter >= WARM_UP) {
	PGI_time = get_time() - start;
	HMC_PGI_time += PGI_time;
	PGI_total_time += PGI_time;
#ifdef TIMING_COMP
	SERIAL_time = get_time() - start;
	HMC_SERIAL_time += SERIAL_time;
	SERIAL_total_time += SERIAL_time;
#endif
      }
    }
  } //END acc region
  
  if(iter%MEAS == 0 && iter >= WARM_UP){
#ifdef TIMING_COMP  
    printf("TIMING       HMC: PGI time %.6e, serial time %.6e, speedup: %.4f\n",
	   HMC_PGI_time, HMC_SERIAL_time, HMC_SERIAL_time/HMC_PGI_time);
#else
    printf("TIMING       HMC: PGI time %.6e\n", HMC_PGI_time);
#endif
  }
  
  return accepted;
}

// Computes the improved correlator via the SW clusters and Fourier
// transforms.
void twoPtMomSpace(const int cluster_num, const double phiCOPY[L][L],
		   const int label[L][L], double twoPtKmode[L][L]) {

  //We indicate that the data is present on the device
#pragma acc kernels present(phiCOPY[0:L][0:L], label[0:L][0:L], twoPtKmode[0:L][0:L])
  {

    const double Linv = 1.0/L;

    //We parallelise over the k modes. This ensures that there are
    //plenty of threads and that each thread has enough work.
#pragma acc loop independent tile(4,4)
    //#pragma acc parallel loop collapse(2)
    for(int kx=0; kx<L; kx++) {
      for(int ky=0; ky< L; ky++) {

	//Everything between these lines is performed by a single thread
	//-------------------------------------------------------------------------
	double temp[2] = {0.0,0.0};	
	//set first cluster
	int cluster = label[0][0];
	
	for(int x=0; x<L; x++) {
	  for(int y=0; y<L; y++) {
	    if(label[x][y] == cluster) {
	      //place data in temp sum
	      temp[0] += phiCOPY[x][y] * cosf(TWO_PI*(x*kx + y*ky)*Linv);
	      temp[1] -= phiCOPY[x][y] * sinf(TWO_PI*(x*kx + y*ky)*Linv);
	    }
	    else {
	      //We have found a new cluster. 
	      //Square the previous result,
	      twoPtKmode[kx][ky] += temp[0]*temp[0] + temp[1]*temp[1];
	      
	      //reset the temp sum,
	      temp[0] =  phiCOPY[x][y] * cosf(TWO_PI*(x*kx + y*ky)*Linv);
	      temp[1] = -phiCOPY[x][y] * sinf(TWO_PI*(x*kx + y*ky)*Linv);
	      
	      //set a new cluster.
	      cluster = label[x][y];
	    }
	  }
	}
	//-------------------------------------------------------------------------
	
      }
    }
  }// END ACC REGION  
}

//Computes the action of the field
double calcH(double mom[L][L], double phi[L][L], param_t p) {

  const double musqr = p.musqr;
  const double lambda = p.lambda;
  
  double Hout = 0.0;
  //We indicate that the data is present on the device
#pragma acc data present(mom[0:L][0:L], phi[0:L][0:L])
  {
    //If we declare a variable inside a parallel region,
    //the compiler will ensure that eath thread has
    //a private copy.
    double H = 0.0;
#pragma acc parallel loop tile(32,4) reduction(+:H)   
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++) {

	/*
	H += 0.5*(phi[x][y]- phi[(x+1)%L][y])*(phi[x][y]- phi[(x+1)%L][y]);
	H += 0.5*(phi[x][y]- phi[x][(y+1)%L])*(phi[x][y]- phi[x][(y+1)%L]);
	H += (-musqr + lambda*phi[x][y]*phi[x][y])*phi[x][y]*phi[x][y];
	H += 0.5 * mom[x][y] * mom[x][y];
	*/
	
	H += (0.5*(phi[x][y]- phi[(x+1)%L][y])*(phi[x][y]- phi[(x+1)%L][y])
	      + 0.5*(phi[x][y]- phi[x][(y+1)%L])*(phi[x][y]- phi[x][(y+1)%L])
	      + (-musqr + lambda*phi[x][y]*phi[x][y])*phi[x][y]*phi[x][y] 
	      +0.5 * mom[x][y] * mom[x][y]);
	
      }   
    Hout = H;
  }
  return Hout;
}

#if 0
void trajectory(double mom[L][L], double phi[L][L], param_t p) {

  const double dt = p.dt;
  const double musqr = p.musqr;
  const double lambda = p.lambda;
  const int nstep = p.nstep;
  
  //We indicate that the data is present on the device
  
  double fU[L][L];
  double dt = p.dt;
#pragma acc data present(mom[0:L][0:L], phi[0:L][0:L]) create(fU[0:L][0:L])
  {
    //Initial half step:
    //P_{1/2} = P_0 - dtau/2 * fU
    forceU(fU, phi, p);
    update_mom(fU, mom, p, 0.5*dt);
    
    //step loop
    for(int k=1; k<p.nstep; k++) {

      //U_{k} = U_{k-1} + P_{k-1/2} * dt
      update_phi(phi, mom, p, dt);

      //P_{k+1/2} = P_{k-1/2} - fU * dt 
      forceU(fU, phi, p);
      update_mom(fU, mom, p, dt);
      
    } //end step loop

    //Final half step.
    //U_{n} = U_{n-1} + P_{n-1/2} * dt
    update_phi(phi, mom, p, dt);
    forceU(fU, phi, p);
    update_mom(fU, mom, p, 0.5*dt);
  }
  return;
}

void forceU(double fU[L][L], double phi[L][L], param_t p) {
  
#pragma acc data present(fU[0:L][0:L], phi[0:L][0:L])
  {
    double dt = p.dt;
    //zeroField(fV);
#pragma acc parallel loop tile(32,2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++) {
	fU[x][y] = 0.0;
	fU[x][y] +=  -2.0*phi[x][y] + phi[(x+1)%L][y] + phi[(x-1+L)%L][y];
	fU[x][y] +=  -2.0*phi[x][y] + phi[x][(y+1)%L] + phi[x][(y-1+L)%L];
	fU[x][y] +=  (2.0*p.musqr - 4.0*p.lambda*phi[x][y]*phi[x][y]) * phi[x][y];
      }
  }
  return;
}
#endif


void update_mom(double mom[L][L], double fU[L][L], param_t p, double dt) {
  
#pragma acc data present(mom[0:L][0:L], fU[0:L][0:L])
  {
#pragma acc parallel loop collapse(2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++) 
	mom[x][y] -= fU[x][y] * dt;
  }
}

void update_phi(double phi[L][L], double mom[L][L], param_t p, double dt) {
  
#pragma acc data present(phi[0:L][0:L], mom[0:L][0:L])
  {
#pragma acc parallel loop collapse(2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++) 
	phi[x][y] += mom[x][y] * dt;
  }
  return
}


void systemRands(double rands[L][L], int block) {
  for(int y=0; y<L; y++)
    rands[block][y] = drand48();   
  return;
}
 
void HMCsystemRands(double *rands, int length) {
  for(int x=0; x<length; x++)
    rands[x] = drand48();   
  return;
}

//Device function
void gaussReal_F(double field[L][L], double rands[L][L]) {
  //normalized gaussian exp[ - phi*phi/2]  <eta^2> = 1
  //We indicate that the data is present on the device
#pragma acc data present(field[0:L][0:L], rands[0:L][0:L])
  {
    double r, theta;
    const double twoPI = TWO_PI;
    //#pragma acc parallel loop collapse(2)
#pragma acc parallel loop tile(32,2)
    for(int x=0; x<L; x++){
      for(int y=0; y<L; y += 2){
	r = sqrt(-2.0*log(rands[x][y]));
	theta = twoPI*rands[x][y+1];
	field[x][y  ] = r*cos(theta);
	field[x][y+1] = r*sin(theta);
      }
    }
  }
  return;
}


//bond[x][y][mu]  mu = 0:x+1  1:y+1, 2:x-1, 3:y-1
//-----------------------------------------------------------------
void LatticePercolate(bool bond[L][L][2*D], int label[L][L], const double phi[L][L],
		      const double rands1[L][L], const double rands2[L][L], const double rands3[L][L]) {
  
#pragma acc data present(bond[0:L][0:L][0:2*D], label[0:L][0:L], phi[0:L][0:L], rands1[0:L][0:L], rands2[0:L][0:L], rands3[0:L][0:L])
  {
    
    //#pragma acc parallel loop tile(32,4) collapse(2)
#pragma acc parallel loop tile(32,2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++){
	
	double probability = 1.0 - exp( -2.0 * phi[x][y]*phi[(x + 1)%L][y]);
	if (rands1[x][y] < probability) {
	  bond[x][y][0] = true;
	  bond[(x+1)%L][y][2] = true;
	}
	else {
	  bond[x][y][0] = false;
	  bond[(x+1)%L][y][2] = false;
	} 
	
	probability = 1.0 - exp( -2.0 * phi[x][y]*phi[x][(y+1)%L]);
	if(rands2[x][y] < probability) {
	  bond[x][y][1] = true;
	  bond[x][(y+1)%L][3] = true;
	}
	else {
	  bond[x][y][1] = false;
	  bond[x][(y+1)%L][3] = false;
	}
	
	// Random spin on labels = p/m (1, 2, ... L*L):
	if(rands3[x][y] < 0.5) label[x][y] = -(y + x*L + 1);
	else label[x][y] =  (y + x*L + 1);
      }
  }  
  return;
}

//Single Grid implemenation at present.
void MultigridSW(int label[L][L], const bool bond[L][L][2*D], int loops) {
  
#pragma acc data present(bond[0:L][0:L][0:2*D], label[0:L][0:L])
  {
    //Use a quirk of the architecture to accelerate the relaxtion procedure.
    //A single relaxtion sweep is defined as a loop over x and y. By introducing
    //a new outer loop, it would appear to perform redundant steps. However, we
    //change the labels in global memory as soon as the thread is complete and the
    //streaming multiprocessors launch in an arbitrary sequence. As a result, the
    //labels are able to propagate.
    //
    //Many of the threads perform rendundant checks. The wall clock latency
    //associated with 'loops' kernel launches is huge compared to the wall clock
    //time wasted by the redundant loops, so the net result is an algorithmically
    //inefficient kernel that gets the results much faster!
#pragma acc parallel loop collapse(3)
    for(int a=0; a<loops; a++) {
      for(int x=0; x<L; x++)
	for(int y=0; y<L; y++) {
	  int minLabel = label[x][y];  // Find min of connection to local 4 point stencil.
	  
	  if( bond[x][y][0] && (abs(minLabel) > abs(label[(x+1)%L][y])) ) {
	    minLabel = label[(x+1)%L][y];
	  }
	  
	  if( bond[x][y][1] && (abs(minLabel) > abs(label[x][(y+1)%L])) ) {
	    minLabel = label[x][(y+1)%L];
	  }
	  
	  if( bond[x][y][2] && (abs(minLabel) > abs(label[(x-1+L)%L][y])) ) {
	    minLabel = label[(x-1+L)%L][y];
	  }
	  
	  if( bond[x][y][3] && (abs(minLabel) > abs(label[x][(y-1+L)%L])) ) {
	    minLabel = label[x][(y-1+L)%L];
	  }
	  
	  label[x][y] = minLabel;
	}
    }//end loops
  }
}


//Put relax inside the Multigrid
//Single Grid implemenation at present.
//-----------------------------------------------------------
bool MultigridSWTest(int label[L][L], const bool bond[L][L][2*D]) {
  
  bool stopRet = true;
  int stopOut = 0;
#pragma acc data present(bond[0:L][0:L][0:2*D], label[0:L][0:L])
  {
    int stop = 0;
#pragma acc parallel loop tile(32,2) reduction(+:stop) 
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++) {
	
	int minLabel = label[x][y];  // Find min of connection to local 4 point stencil.
	
	if( bond[x][y][0] && (abs(minLabel) > abs(label[(x+1)%L][y])) ) {
	  minLabel = label[(x+1)%L][y];
	  stop++;
	}
	
	if( bond[x][y][1] && (abs(minLabel) > abs(label[x][(y+1)%L])) ) {
	  minLabel = label[x][(y+1)%L];
	  stop++;
	}
	
	if( bond[x][y][2] && (abs(minLabel) > abs(label[(x -1 + L)%L][y])) ) {
	  minLabel = label[(x-1+L)%L][y];
	  stop++;
	}
	
	if( bond[x][y][3] && (abs(minLabel) > abs(label[x][(y -1 + L)%L])) ) {
	  minLabel = label[x][(y-1+L)%L];
	  stop++;
	}	
	label[x][y] = minLabel;
      }
    
    stopOut = stop;
    
  }
  
  //printf("Stop = %d\n", stopOut);
  if(stopOut > 0) stopRet = false;
  return stopRet;
  
}


void FlipSpins(double phi[L][L], const int label[L][L]) {
#pragma acc data present(phi[0:L][0:L], label[0:L][0:L])
  {
#pragma acc parallel loop collapse(2)
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++)
	if(label[x][y] < 0) phi[x][y] *= -1.0;
  }
  return;
}


// Begin SERIAL routines
//-------------------------------------------------------------------------------------------

double calcH_serial(double mom[L][L], double phi[L][L],  param_t p) {

  double H = 0.0;
  int xp1;
  int yp1;
  for(int x=0; x<L; x++) {
    int xp1 = (x+1)%L;
    for(int y=0; y<L; y++) {
      int yp1 = (y+1)%L;
      
      H += 0.5*(phi[x][y] - phi[xp1][y])*(phi[x][y]- phi[xp1][y]);
      H += 0.5*(phi[x][y] - phi[x][yp1])*(phi[x][y]- phi[x][yp1]);
      H += (-p.musqr + p.lambda*phi[x][y]*phi[x][y])*phi[x][y]*phi[x][y];
      H += mom[x][y] * mom[x][y]/2.0;
      
    }
  }
  return H;
}

void trajectory_serial(double mom[L][L], double phi[L][L], param_t p) {

  int xp1, xm1, yp1, ym1;
  
  //Initial half step
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      phi[x][y] += mom[x][y] * p.dt/2.0;
  
  //Steps
<<<<<<< HEAD
  for(int k=1; k<p.nstep; k++) {      
    for(int x=0; x<L; x++)
      for(int y=0; y<L; y++)
	phi[x][y] += mom[x][y] * p.dt;
=======
  for(int step = 1; step < p.nstep; step++) {      
    for(int x =0;x< L;x++)
      for(int y =0;y< L;y++)
	phi[x][y] +=   mom[x][y] * p.dt;
    
    for(int x=0; x<L; x++) {
      xp1 = (x+1)%L;
      xm1 = (x-1+L)%L;
      for(int y=0; y<L; y++) {
	yp1 = (y+1)%L;
	ym1 = (y-1+L)%L;
	
	mom[x][y] += (-2.0*phi[x][y] + phi[xp1][y] + phi[xm1][y])* p.dt;
	mom[x][y] += (-2.0*phi[x][y] + phi[x][yp1] + phi[x][ym1])* p.dt;
	mom[x][y] += ( 2.0*p.musqr - 4.0*p.lambda*phi[x][y]*phi[x][y] ) * phi[x][y] * p.dt;
      }
    }
  } //end step loop                                                                                               
  
  //Final half step
  for(int x =0;x< L;x++)
    for(int y =0;y< L;y++)
      phi[x][y] += mom[x][y] * p.dt/2.0;
  
  return;
}

 
 
void LatticePercolate_serial(bool bond[L][L][2*D], int label[L][L], const double phi[L][L]){
    
  double probability;
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++){
      probability = 1.0 -  exp( -2.0 * phi[x][y]*phi[(x + 1)%L][y]);
      if (drand48() < probability) {
	bond[x][y][0] = true;
	bond[(x+1)%L][y][2] = true;
      }
      else {
	bond[x][y][0] = false;
	bond[(x+1)%L][y][2] = false;
      }
      probability = 1.0 -  exp( -2.0 * phi[x][y]*phi[x][(y+1)%L]);
      if(drand48() < probability) {
	bond[x][y][1] = true;
	bond[x][(y+1)%L][3] = true;
      }
      else {
	bond[x][y][1] = false;
	bond[x][(y+1)%L][3] = false;
      }
      
      // Random spin on labels = p/m (1, 2, ... L*L):
      if(drand48() < 0.5) label[x][y] = -(y + x*L + 1);
      else label[x][y] =  (y + x*L + 1);
    }
  
  return;
}


bool MultigridSW_serial(int label[L][L],const bool bond[L][L][2*D]) {

  bool stop = true;
  int minLabel;
  
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++) {
      minLabel = label[x][y];  // Find min of connection to local 4 point stencil.
      
      if( bond[x][y][0] && (abs(minLabel) > abs(label[(x+1)%L][y])) ) {
	minLabel = label[(x+1)%L][y];
	stop = false;
      }
      
      if( bond[x][y][1] &&  (abs(minLabel) > abs(label[x][(y+1)%L])) ) {
	minLabel = label[x][(y+1)%L];
	stop = false;
      }
      
      if( bond[x][y][2] && (abs(minLabel) > abs(label[(x-1+L)%L][y])) ) {
	minLabel = label[(x-1+L)%L][y];
	stop = false;
      }
      
      if( bond[x][y][3] && (abs(minLabel) > abs(label[x][(y-1+L)%L])) ) {
	minLabel = label[x][(y-1+L)%L];
	stop = false;
      }
      
      label[x][y] = minLabel;
    }
  
  return stop;
}

void FlipSpins_serial(double phi[L][L], const int label[L][L]) {
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      if(label[x][y] < 0) phi[x][y] *= -1.0;
  return;
}

//Host function
void gaussReal_F_serial(double field[L][L], double rands[L][L]) {
  //normalized gaussian exp[ - phi*phi/2]  <eta^2> = 1
  double r, theta;
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y+=2){
      r = sqrt(-2.0*log(rands[x][y]));
      theta = TWO_PI*rands[x][y+1];
      field[x][y  ] = r*cos(theta);
      field[x][y+1] = r*sin(theta);
    }
  return;
}

void twoPtMomSpace_serial(const int cluster_num, const double phiCOPY[L][L],
			  const int label[L][L], double twoPtKmode[L][L]) {
  
  //Quick and dirty 2D FT.
  const double Linv = 1.0/L;
  complex<double> twopt_cluster_sum[cluster_num];
  for(int kx=0; kx<L; kx++) {
    for(int ky=0; ky<L; ky++) {
      for(int a=0; a<cluster_num; a++) twopt_cluster_sum[a] = 0.0;
      
      //Loop over all spatial sites, collect cluster sums
      int coor = 0;
      for(int x=0; x<L; x++) {
	for(int y=0; y<L; y++) {
	  twopt_cluster_sum[label[x][y] - 1] +=
	    phiCOPY[x][y] * (cos(TWO_PI*(x*kx + y*ky)*Linv) -
			     I*sin(TWO_PI*(x*kx + y*ky)*Linv));
	  coor++;
	}
      }
      
      //The k mode is complete. Now we may sum each contribution and square
      for(int a=0; a<cluster_num; a++) {
	twoPtKmode[kx][ky] +=
	  (twopt_cluster_sum[a].real() * twopt_cluster_sum[a].real() +
	   twopt_cluster_sum[a].imag() * twopt_cluster_sum[a].imag() );
      }
    }
  }
}

// Measure the average phi field value
double measMag_serial(const double phi[L][L]) {

  double mag = 0.0;
  for(int x=0; x<L; x++)
    for(int y=0; y<L; y++)
      mag += phi[x][y];
  
  return mag;
}


// End SERIAL routines
//-----------------------------------------------------------------------------------
