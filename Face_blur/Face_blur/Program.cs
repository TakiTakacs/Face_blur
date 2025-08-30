using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;




namespace Face_blur
{
    class Program
    {
        static void Main(string[] args)
        {
            //haarcascade file-ok megadása felismeréshez
            CascadeClassifier faceHaarCascade,face2HaarCascade;
            faceHaarCascade = new CascadeClassifier(@"haarcascade_frontalface_default.xml");
            face2HaarCascade = new CascadeClassifier(@"haarcascade_frontalface_alt.xml");

            //MSMF backend használata hogy ne kelljen külön ffmpegnek telepítve lennie
            Backend[] backends = CvInvoke.WriterBackends;
            int backend_idx = 0; //bármelyik backend
            foreach (Backend be in backends)
            {
                if (be.Name.Equals("MSMF"))
                {
                    backend_idx = be.ID;
                    break;
                }
            }

            //videó capture
            using (var capture = new VideoCapture(@"test1.mp4"))
            {

                //Video exportáláshoz paraméterek
                int width = (int)capture.Get(CapProp.FrameWidth);
                int height = (int)capture.Get(CapProp.FrameHeight);
                double fps = capture.Get(CapProp.Fps);

                using (var vw = new VideoWriter("output.mp4",backend_idx, VideoWriter.Fourcc('H', '2', '6', '4'), fps, new Size(width, height), true))
                {

                
                //ideiglenes képkocka számláló progress helyett
                int frameIndex = 0;
                Mat frame = new Mat();

                    //képkockák feldolgozása
                    while (capture.Grab())
                    {

                        capture.Retrieve(frame);

                        if (!frame.IsEmpty)
                        {
                            var frameImg = frame.ToImage<Bgr, byte>();

                            var gray = frame.ToImage<Gray, byte>();

                            //arcok felismerése és bekeretezése
                            Rectangle[] jelenArcok = faceHaarCascade.DetectMultiScale(gray, 1.2);
                            Rectangle[] jelenArcok2 = face2HaarCascade.DetectMultiScale(gray, 1.2);

                            foreach (var item in jelenArcok2)
                            {
                                frameImg.Draw(item, new Bgr(255, 0, 0), 4);
                            }

                            foreach (var arc in jelenArcok)
                            {

                                frameImg.Draw(arc, new Bgr(0, 0, 255), 4);

                            }


                            //képkocka file-ba írása
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
}
