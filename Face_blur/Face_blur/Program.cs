using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using System.Drawing;
using System.IO;




namespace Face_blur
{
    class Program
    {

        static void Main(string[] args)
        {

            var smallframe = new Mat();
            //haarcascade file-ok megadása felismeréshez
            
            using var faceHaarCascade = new CascadeClassifier(@"haarcascade_frontalface_default.xml");
            using var face2HaarCascade = new CascadeClassifier(@"haarcascade_frontalface_alt.xml");

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
            using var capture = new VideoCapture(@"test3.mp4");
            

            //Video exportáláshoz paraméterek
            int width = (int)capture.Get(CapProp.FrameWidth);
            int height = (int)capture.Get(CapProp.FrameHeight);
            double fps = capture.Get(CapProp.Fps);


            using var vw = new VideoWriter("output.mp4", backend_idx, VideoWriter.Fourcc('H', '2', '6', '4'), fps, new Size(width, height), true);
                

                
                //ideiglenes képkocka számláló progress helyett
                int frameIndex = 0;
                Mat frame = new Mat();

                //képkockák feldolgozása                
                while (capture.Grab())
                {

                    capture.Retrieve(frame);

                    if (!frame.IsEmpty)
                    {
                       using var frameImg = frame.ToImage<Bgr, byte>();

                       using var gray = frame.ToImage<Gray, byte>();

                        //arcok felismerése és bekeretezése
                        Rectangle[] jelenArcok = face2HaarCascade.DetectMultiScale(gray, 1.1,4);

                        foreach (var item in jelenArcok)
                        {
                            frameImg.Draw(item, new Bgr(255, 0, 0), 4);


                            int extraPixel = 50; 

                            // Calculate expanded boundaries with safety checks
                            int x = Math.Max(item.Left - extraPixel, 0);
                            int y = Math.Max(item.Top - extraPixel, 0);
                            int right = Math.Min(item.Right + extraPixel, frameImg.Width);
                            int bottom = Math.Min(item.Bottom + extraPixel, frameImg.Height);
                            int widt = right - x;
                            int heigh = bottom - y;

                            // Only process if the ROI is valid
                            if (width > 0 && height > 0)
                            {
                                using var faceImg = new Mat(frameImg.Mat, new Rectangle(x, y, widt, heigh));
                                using var faceBlur = new Mat();
                                CvInvoke.GaussianBlur(faceImg, faceBlur, new Size(51, 51), 30);

                                // Copy the blurred face back to the original image
                                faceBlur.CopyTo(new Mat(frameImg.Mat, new Rectangle(x, y, widt, heigh)));
                            }

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
