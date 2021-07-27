#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  cout << "Yo, OpenCV version : " << CV_VERSION << endl;

  if ( CV_MAJOR_VERSION < 3)
  {
      // Old OpenCV 2 code goes here.
  } else
  {
      // New OpenCV 3 code goes here.
  }
}