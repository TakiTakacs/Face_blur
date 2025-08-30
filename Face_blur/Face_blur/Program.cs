using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.Util;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Windows;




namespace Face_blur
{
    class Program
    {
        static void Main(string[] args)
        {
            
            CascadeClassifier haarCascade;
            haarCascade = new CascadeClassifier(@"haarcascade_frontalface_alt_tree.xml");

            //Image<Bgr, Byte> img = new Image<Bgr, byte>();

            //var asd = My_Image.Convert<Gray, byte>();

            //asd.Save("output.jpg");

            using (var capture = new VideoCapture(@"test1.mp4"))
            {
                int frameIndex = 0;
                Mat frame = new Mat();
                //List<Rectangle> rectangles = new List<Rectangle>();

                while (capture.Grab())
                {
                    capture.Retrieve(frame); // Get frame

                    if (!frame.IsEmpty)
                    {
                        var gray = frame.ToImage<Gray,byte>();

                        var grayBitmap = gray.ToBitmap();

                        Bitmap tempBitmap = new Bitmap(grayBitmap.Width,grayBitmap.Height);

                        var jelenArcok = haarCascade.DetectMultiScale(gray);

                        foreach ( var arc in jelenArcok)
                        {
                            using (Graphics graphics = Graphics.FromImage(tempBitmap))
                            {
                                graphics.DrawImage(grayBitmap,0,0);
                                using (Pen pen = new Pen(Color.Red, 1))
                                {
                                    graphics.DrawRectangle(pen, arc);
                                }
                            }
                        }
                        grayBitmap = tempBitmap;
                        

                        grayBitmap.Save(@$"C:\Users\peter\source\repos\Face_blur\Face_blur\Face_blur\bin\Debug\net9.0\Frames\frame_{frameIndex}.jpg");
                        Console.WriteLine($"Saved");
                        frameIndex++;
                    }
                }



            }




                



        }

      

    }
}
