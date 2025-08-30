using Emgu.CV;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Windows;
using Emgu.Util;
using System.Drawing.Imaging;




namespace Face_blur
{
    class Program
    {
        static void Main(string[] args)
        {
            
            CascadeClassifier haarCascade;


            //var video = File.Open("/home/taki04/Programozás/Face_blur/testvideos/test1.mp4", FileMode.Open);

            haarCascade = new CascadeClassifier(@"haarcascade_frontalface_alt_tree.xml");

            Image<Bgr, Byte> My_Image = new Image<Bgr, byte>(@"test1.jpg");

            var asd = My_Image.Convert<Gray, byte>();

            asd.Save("output.jpg");



        }

      

    }
}
