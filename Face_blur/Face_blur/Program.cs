using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;




namespace Face_blur
{
    class Program
    {
        static void Main(string[] args)
        {
            
            CascadeClassifier faceHaarCascade,face2HaarCascade;
            faceHaarCascade = new CascadeClassifier(@"haarcascade_frontalface_default.xml");
            face2HaarCascade = new CascadeClassifier(@"haarcascade_frontalface_alt.xml");
            var vw = new VideoWriter("output.avi", 30, new Size(2160, 3840), true);


            using (var capture = new VideoCapture(@"test3.mp4"))
            {
                


                int frameIndex = 0;
                Mat frame = new Mat();

                while (capture.Grab())
                {
                    capture.Retrieve(frame); 

                    if (!frame.IsEmpty)
                    {
                        var frameImg = frame.ToImage<Rgb, byte>();

                        var gray = frame.ToImage<Gray,byte>();


                        Rectangle[] jelenArcok = faceHaarCascade.DetectMultiScale(gray,1.2);
                        Rectangle[] jelenSzemek = face2HaarCascade.DetectMultiScale(gray, 1.2);

                        foreach (var item in jelenSzemek)
                        {
                            frameImg.Draw(item, new Rgb(255, 0, 0), 4);
                        }

                        foreach ( var arc in jelenArcok)
                        {

                            frameImg.Draw(arc,new Rgb(0,0,255) ,4);

                        }

                        
                        
                        vw.Write(frameImg);

                        //frameImg.Save(@$"C:\Users\peter\source\repos\Face_blur\Face_blur\Face_blur\bin\Debug\net9.0\Frames\frame_{frameIndex}.jpg");
                        Console.WriteLine($"Saved frame: {frameIndex}");
                        frameIndex++;
                    }

                }
            }
        }     
    }
}
