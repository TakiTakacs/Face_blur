using OpenCvSharp;
using OpenCvSharp.Dnn;




namespace Face_blur
{
    class Program
    {

        static void Main(string[] args)
        {
            int progress = 0;
            string inputFilePath = @"C:\Users\peter\Videos\test1.mp4";
            string outputFilePath = @"C:\Users\peter\Videos\output2.mp4";
            Console.WriteLine($"Progress: {progress}%");

            //argumentumok feldolgozása
            if (args.Length == 2)
            {
                inputFilePath = args[0];
                outputFilePath = args[1];
            }

            //DNN model betöltése
            using var net = CvDnn.ReadNetFromCaffe("deploy.prototxt", "dnn_model.caffemodel");

            

            //net.SetPreferableBackend(Backend.OPENCV);
            //net.SetPreferableTarget(Target.OPENCL);

            //videó capture
            using var capture = new VideoCapture(inputFilePath);

            

            //Video exportáláshoz paraméterek
            int width = capture.FrameWidth;
            int height = capture.FrameHeight;
            double fps = capture.Fps;
            var frameCount = capture.FrameCount;


            using var vw = new VideoWriter(outputFilePath,VideoCaptureAPIs.Any, VideoWriter.FourCC('m', 'p', '4', 'v'), fps, new Size(width, height), true);
                

                
                //ideiglenes képkocka számláló progress helyett
                int frameIndex = 0;
                Mat frame = new Mat();


                //Pontatlanság okozta flicker elkerülése miatt, korábban maszkolt területek további maszkolása adott képkoca ideig
                //const int maxFramePersistence = 15; //Arc helyének takarása extra képkockákig
                //Dictionary<Rectangle, int> facePersistence = new Dictionary<Rectangle, int>();
                

            //képkockák feldolgozása                
            while (capture.Grab())
                {

                    capture.Retrieve(frame);

                    if (!frame.Empty())
                    {
                       using var blob = CvDnn.BlobFromImage(frame,1.0,new Size(2160,3840),new Scalar(104.0,117.0,123.0),false,false);

                        net.SetInput(blob,"data");

                    using var detection = net.Forward("detection_out");
                    using var detectionMat = Mat.FromPixelData(detection.Size(2),detection.Size(3),MatType.CV_32F,detection.Ptr(0));

                    for (int i = 0; i < detectionMat.Rows; i++)
                    {
                        float confidence = detectionMat.At<float>(i, 2);
                        if (confidence > 0.7)
                        {
                            int x1 = (int)(detectionMat.At<float>(i, 3) * width);
                            int y1 = (int)(detectionMat.At<float>(i, 4) * height);

                            int x2 = (int)(detectionMat.At<float>(i, 5) * width);
                            int y2 = (int)(detectionMat.At<float>(i, 6) * height);

                            Cv2.Rectangle(frame, new Point(x1, y1), new Point(x2, y2), Scalar.Green);

                            if (y1>=height)
                            {
                                y1 = height-1;
                            }
                            else if (y1<1)
                            {
                                y1 = 1;
                            }
                            if (y2 >= height)
                            {
                                y2 = height;
                            }
                            if (x2 >= width)
                            {
                                x2 = width;
                            }
                            if (x1 >= width)
                            {
                                x1 = width-1;
                            }
                            else if (x1 < 1)
                            {
                                x1 = 1;
                            }

                            using var faceImg = new Mat(frame, new OpenCvSharp.Range(y1, y2), new OpenCvSharp.Range(x1, x2));

                            using var faceBlur = new Mat();
                            Cv2.GaussianBlur(faceImg, faceBlur,new Size(31,31),30);
                            frame[new OpenCvSharp.Range(y1, y2), new OpenCvSharp.Range(x1, x2)] = faceBlur;

                        }
                    }


                    //képkocka file-ba írása
                    vw.Write(frame);

                    //progress nyomon követése
                    var most = Convert.ToInt32(Math.Round(Convert.ToDouble((frameIndex + 1) / frameCount * 100)));
                    //frameImg.Save(@$"C:\Users\peter\source\repos\Face_blur\Face_blur\Face_blur\bin\Debug\net9.0\Frames\frame_{frameIndex}.jpg");
                    //if (progress < most)
                    //{
                        progress = most;
                        Console.Clear();
                        Console.WriteLine($"Progress: {frameIndex+1}/{frameCount}");
                    //}

                        frameIndex++;
                    }

                }           
        }     
    }
}
