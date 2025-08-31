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

            string inputFilePath = @"test4.mp4";
            string outputFilePath = "output.mp4";
            if (args.Length == 2)
            {
                inputFilePath = args[0];
                outputFilePath = args[1];
            }


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
            using var capture = new VideoCapture(inputFilePath);

            

            //Video exportáláshoz paraméterek
            int width = (int)capture.Get(CapProp.FrameWidth);
            int height = (int)capture.Get(CapProp.FrameHeight);
            double fps = capture.Get(CapProp.Fps);


            using var vw = new VideoWriter(outputFilePath, backend_idx, VideoWriter.Fourcc('H', '2', '6', '4'), fps, new Size(width, height), true);
                

                
                //ideiglenes képkocka számláló progress helyett
                int frameIndex = 0;
                Mat frame = new Mat();


                //Pontatlanság okozta flicker elkerülése miatt, korábban maszkolt területek további maszkolása adott képkoca ideig
                const int maxFramePersistence = 5; //Arc helyének takarása extra képkockákig
                Dictionary<Rectangle, int> facePersistence = new Dictionary<Rectangle, int>();
                

            //képkockák feldolgozása                
            while (capture.Grab() && frameIndex<1000)
                {

                    capture.Retrieve(frame);

                    if (!frame.IsEmpty)
                    {
                       //frame Image-é konvertálása
                       using var frameImg = frame.ToImage<Bgr, byte>();
                        
                       //grayscale a felismeréshez
                       using var gray = frame.ToImage<Gray, byte>();

                        //arcok felismerése és bekeretezése
                        Rectangle[] jelenArcok = face2HaarCascade.DetectMultiScale(gray, 1.1,4);

                        var currentFaces = new List<Rectangle>(jelenArcok);
                     
                        //korábbról mentett arcok hátralevő megjelenési idejének csökkentése, vagy lejártak törlése
                        foreach (var face in facePersistence.Keys)
                        {
                            facePersistence[face]--;
                            if (facePersistence[face] <= 0)
                            {
                                facePersistence.Remove(face);
                            }
                        }


                    //összes feldolgozandó arc egybevonása és uj arcok hozzáadása a facePersistence listához
                    var allFaces = facePersistence.Keys.ToList();

                    for (int i = 0; i < jelenArcok.Length; i++)
                    {
                        allFaces.Add(jelenArcok[i]);
                        if (!facePersistence.ContainsKey(jelenArcok[i]))
                        {
                            facePersistence.Add(jelenArcok[i], maxFramePersistence);                            
                        }
                    }

                    //arcok maszkolása
                    foreach (var item in allFaces)
                    {

                        int extraPixel = 40;

                        //maszkolás határainak ellenőrzése
                        int x = Math.Max(item.Left - extraPixel, 0);
                        int y = Math.Max(item.Top - extraPixel, 0);
                        int right = Math.Min(item.Right + extraPixel, frameImg.Width);
                        int bottom = Math.Min(item.Bottom + extraPixel, frameImg.Height);
                        int widt = right - x;
                        int heigh = bottom - y;

                        //megfelelő helyek maszkolása
                        if (width > 0 && height > 0)
                        {
                            using var faceImg = new Mat(frameImg.Mat, new Rectangle(x, y, widt, heigh));
                            using var faceBlur = new Mat();
                            CvInvoke.GaussianBlur(faceImg, faceBlur, new Size(51, 51), 30);

                            //maszkolt kép másolása az eredetire
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
