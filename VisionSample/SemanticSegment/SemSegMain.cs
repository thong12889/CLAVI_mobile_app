using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace VisionSample
{
    public class SemSegMain : VisionSampleBase<SemSegImageProcessor>
    {
        public const string Identifier = "Semantic Segment";
        public const string ModelFilename = "SemSeg.onnx";

        public SemSegMain()
            : base(Identifier, ModelFilename) { }

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplySemanticToImage(predictions, sourceImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        SKBitmap GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight)
        {
            // Setup inputs and outputs
            var inputMeta = Session.InputMetadata;
            var inputName = inputMeta.Keys.ToArray()[0];
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };

            // Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            //Postprocessing
            var resultsArray = results.ToArray();
            //Pred
            var pred_value = resultsArray[0].AsEnumerable<Int64>().ToArray();
            var pred_dim = resultsArray[0].AsTensor<Int64>().Dimensions.ToArray();

            var output = GetSemanticMask(pred_value, pred_dim, SemSegLabelMap.Labels.Length);
            output = output.Resize(new SKImageInfo(sourceImageWidth, sourceImageHeight), SKFilterQuality.High);

            return output;
        }

        SKBitmap GetSemanticMask(long[] output_value, int[] output_dim, int label_num)
        {
            var pallette = GenPalette(label_num);
            pallette[0] = new SKColor(0, 0, 0, 100);
            SKBitmap skBitmap = new SKBitmap(output_dim[3], output_dim[2]);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int h = 0; h < output_dim[2]; h++)
                {
                    for (int w = 0; w < output_dim[3]; w++)
                    {
                        int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                        skBitmap.SetPixel(w, h, pallette[output_value[idx]]);
                    }
                }
            }
            return skBitmap;
        }
        static SKColor[] GenPalette(int classes)
        {
            Random rnd = new Random(classes);
            SKColor[] colors = new SKColor[classes];
            for (int i = 0; i < classes; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                colors[i] = new SKColor(v1, v2, v3, 100);
            }
            return colors;
        }
    }
}
