using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SkiaSharp;

namespace VisionSample
{
    public class AnomalyMain : VisionSampleBase<AnomalyImageProcessor>
    {
        public const string Identifier = "Anomaly";
        public const string ModelFilename = "Anomaly.onnx";

        public AnomalyMain()
            : base(Identifier, ModelFilename) { }

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplyAnomalyToImage(predictions, sourceImage)).ConfigureAwait(false);

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

            var resultsArray = results.ToArray();
            var map_value = resultsArray[0].AsEnumerable<float>().ToArray();
            var map_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
            var score_value = resultsArray[1].AsEnumerable<float>().ToArray();
            var score_dim = resultsArray[1].AsTensor<float>().Dimensions.ToArray();

            var result = GetResultMaskSK(map_value, map_dim, score_value[0]);
            result = result.Resize(new SKImageInfo(sourceImageWidth, sourceImageHeight), SKFilterQuality.None);
            return result;
        }

        SKBitmap GetResultMaskSK(float[] output_value, int[] output_dim, float output2_value)
        {
            SKBitmap skBitmap = new SKBitmap(output_dim[2], output_dim[3]);
            for (int batch = 0; batch < output_dim[0]; batch++)
            {
                for (int cls = 0; cls < output_dim[1]; cls++)
                {
                    for (int h = 0; h < output_dim[2]; h++)
                    {
                        for (int w = 0; w < output_dim[3]; w++)
                        {
                            int idx = (batch * output_dim[1] * output_dim[2] * output_dim[3]) + (cls * output_dim[2] * output_dim[3]) + (h * output_dim[3]) + w;

                            if (output_value[idx] < output2_value)
                            {
                                skBitmap.SetPixel(w, h, new SKColor(0, 0, 0, 50));
                            }
                            else
                            {
                                skBitmap.SetPixel(w, h, new SKColor(255, 0, 0, 50));
                            }
                        }
                    }
                }
            }
            return skBitmap;
        }
    }
}
