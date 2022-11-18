using System.Collections.Generic;
using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class ObjectDetectionMain : VisionSampleBase<ObjImageProcessor>
    {
        public const string Identifier = "Object Detection";
        public const string ModelFilename = "Object.onnx";

        public ObjectDetectionMain()
            : base(Identifier, ModelFilename) { }

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image)
        {
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height, preprocessedImage.Width, preprocessedImage.Height)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplyObjToImage(predictions, sourceImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<ObjPrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight, int inputW, int inputH)
        {
            float threshold = 0.6f;

            // Setup inputs and outputs
            var inputMeta = Session.InputMetadata;
            var inputName = inputMeta.Keys.ToArray()[0];
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, input) };

            //// Run inference
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = Session.Run(inputs);

            // Postprocess
            var resultsArray = results.ToArray();
            //Pred
            var pred_value = resultsArray[0].AsEnumerable<float>().ToArray();
            var pred_dim = resultsArray[0].AsTensor<float>().Dimensions.ToArray();
            //Label
            var label_value = resultsArray[1].AsEnumerable<Int64>().ToArray();

            //Fillter by score
            List<ObjPrediction> objResults = GetObjResults(sourceImageWidth, sourceImageHeight, inputW, inputH, pred_value, pred_dim, label_value, threshold);

            return objResults;
        }

        List<ObjPrediction> GetObjResults(int sourceW, int sourceH, int inputW, int inputH, float[] preds, int[] pred_dim, long[] labels, float pred_thresh = 0.25f)
        {
            List<ObjPrediction> candidates = new List<ObjPrediction>();
            float dw = (float)sourceW / (float)inputW;
            float dh = (float)sourceH / (float)inputH;
            for (int batch = 0; batch < pred_dim[0]; batch++)
            {
                for (int cand = 0; cand < pred_dim[1]; cand++)
                {
                    int score = 4;//Default 4  // object ness score
                    int idx1 = (batch * pred_dim[1] * pred_dim[2]) + cand * pred_dim[2];
                    int idx2 = idx1 + score;
                    var value = preds[idx2];
                    if (value > pred_thresh)
                    {
                        candidates.Add(new ObjPrediction
                        {
                            Box = new PredictionBox(
                                preds[idx1 + 0] * dw,
                                preds[idx1 + 1] * dh,
                                preds[idx1 + 2] * dw,
                                preds[idx1 + 3] * dh),
                            Confidence = preds[idx1 + 4],
                            Label = ObjLabelMap.Labels[labels[idx1 / 5]]
                        });
                    }
                }
            }
            return candidates;
        }
    }
}
