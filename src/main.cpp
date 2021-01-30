#include <iostream>
#include <cmath>
#include <functional>

#include <Eigen/Core>


double distSphere(double x, double y, double z) {
  return 0.5 - std::sqrt(x*x+y*y+z*z);
}

double distCube(double x, double y, double z) {
  double p1 = -x+0.5;
  double p2 = x+0.5;
  double p3 = -y+0.5;
  double p4 = y+0.5;
  double p5 = -z+0.5;
  double p6 = z+0.5;
  return std::min(std::min(std::min(p1,p2), std::min(p3,p4)),std::min(p5,p6));
}

double smoothStep(double x) {
  return x*x * (3.0 - 2.0*x);
}

double clamp(double x, double l, double u) {
  if (x<l) return l;
  if (x>u) return u;
  return x;
}

// compute the surface loss on the grid
double surfaceLoss(const Eigen::MatrixXd& G, 
  std::function<double(double,double,double)> da, 
  std::function<double(double,double,double)> db, 
  double gridSpacing)
{
  double loss = 0.0;
  double gridSpacing2 = 4.0*gridSpacing*gridSpacing;
  
  int rows = G.rows();
  for (int i=0; i<rows; ++i) {
    double x = G(i,0);
    double y = G(i,1);
    double z = G(i,2);
    double dist = da(x,y,z);
    double dist2 = dist*dist;
    double tgtDist = db(x,y,z);
    double tgtDist2 = tgtDist*tgtDist;
    double occupancy = 1.0 - clamp(dist2/gridSpacing2, 0.0, 1.0);
    occupancy = smoothStep(occupancy);
    double tgtOccupancy = 1.0 - clamp(tgtDist2/gridSpacing2, 0.0, 1.0);
    tgtOccupancy = smoothStep(tgtOccupancy);
    loss += (tgtOccupancy*dist2 + occupancy*tgtDist2);
  }

  return loss / double(rows);
}

// compute the alignment loss on the grid
double alignmentLoss(const Eigen::MatrixXd& G, 
  std::function<double(double,double,double)> da, 
  std::function<double(double,double,double)> db, 
  double gridSpacing)
{
  double loss = 0.0;
  double oneoverh = 1.0/gridSpacing;
    
  int rows = G.rows();
  for (int i=0; i<rows; ++i) {
    double x=G(i,0);
    double y=G(i,1);
    double z=G(i,2);
    double daxyz = da(x,y,z);
    double daxhyz = da(x+gridSpacing,y,z);
    double daxyhz = da(x,y+gridSpacing,z);
    double daxyzh = da(x,y,z+gridSpacing);
    Eigen::Vector3d alignment(oneoverh*(daxhyz-daxyz), oneoverh*(daxyhz-daxyz), oneoverh*(daxyzh-daxyz));
    alignment.normalize();
    double dbxyz = db(x,y,z);
    double dbxhyz = db(x+gridSpacing,y,z);
    double dbxyhz = db(x,y+gridSpacing,z);
    double dbxyzh = db(x,y,z+gridSpacing);
    Eigen::Vector3d tgtAlignment(oneoverh*(dbxhyz-dbxyz), oneoverh*(dbxyhz-dbxyz), oneoverh*(dbxyzh-dbxyz));
    tgtAlignment.normalize();
    double dt = alignment.adjoint()*tgtAlignment;
    if (dt*dt > 1e-10)  
      loss += (1-dt*dt);
  }

  return loss / double(rows);
}

double totalLoss(const Eigen::MatrixXd& G, 
  std::function<double(double,double,double)> da, 
  std::function<double(double,double,double)> db, 
  double wSurface, double wAlignment, double gridSpacing)
{
  double sloss = surfaceLoss(G, da, db, gridSpacing);
  double aloss = alignmentLoss(G, da, db, gridSpacing);
  return wSurface*sloss + wAlignment*aloss;
}

int main(int argc, char * argv[])
{
  using namespace Eigen;
  using namespace std;

  // number of vertices on the largest side
  const int s = 64;
  const RowVector3d Vmin(-1.,-1.,-1.);
  const RowVector3d Vmax(1.,1.,1.);

  const double h = (Vmax-Vmin).maxCoeff()/(double)s;
  const RowVector3i res = (s*((Vmax-Vmin)/(Vmax-Vmin).maxCoeff())).cast<int>();
  
  // create grid
  MatrixXd G(res(0)*res(1)*res(2),3);
  for(int zi = 0;zi<res(2);zi++)
  {
    const auto lerp = [&](const int di, const int d)->double
      {return Vmin(d)+(double)di/(double)(res(d)-1)*(Vmax(d)-Vmin(d));};
    const double z = lerp(zi,2);
    for(int yi = 0;yi<res(1);yi++)
    {
      const double y = lerp(yi,1);
      for(int xi = 0;xi<res(0);xi++)
      {
        const double x = lerp(xi,0);
        G.row(xi+res(0)*(yi + res(1)*zi)) = RowVector3d(x,y,z);
      }
    }
  }

  double wsurf = 1.0;
  double walign = 0.01;
  double loss = totalLoss(G, distSphere, distCube, wsurf, walign, h);
  std::cout << "Total loss between unit sphere and cube: " << loss << std::endl;

  loss = totalLoss(G, distSphere, distSphere, wsurf, walign, h);
  std::cout << "Total loss between unit sphere and itself: " << loss << std::endl;

  loss = totalLoss(G, distCube, distCube, wsurf, walign, h);
  std::cout << "Total loss between unit cube and itself: " << loss << std::endl;

  loss = totalLoss(G, distCube, distSphere, wsurf, walign, h);
  std::cout << "Total loss between unit cube and unit sphere: " << loss << std::endl;

  return 0;
}

