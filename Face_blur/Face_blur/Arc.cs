using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace Face_blur
{
    public class Arc
    {
        public Mat mat;
        public int x1,x2,y1,y2;
        public Arc(Mat mat, int x1,int x2,int y1,int y2)
        {
            this.mat = mat;
            this.x1 = x1;
            this.x2 = x2;
            this.y1 = y1;
            this.y2 = y2;
        }
    }
}
