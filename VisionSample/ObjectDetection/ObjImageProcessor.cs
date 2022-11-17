using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class ObjImageProcessor : SkiaSharpImageProcessor<ObjPrediction, float>
    {
        const int RequiredWidth = 640;
        const int RequiredHeight = 640;

        protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
            => sourceImage.Resize(new SKImageInfo(RequiredWidth, RequiredHeight), SKFilterQuality.None);

        protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
        {
            var bytes = image.GetPixelSpan();

            // For the Tensor, we need 3 channels so 320 x 240 x 3 (in RGB format)
            var expectedOutputLength = RequiredWidth * RequiredHeight * 3;

            // The channelData array is expected to be in RGB order with a mean applied i.e. (pixel - 127.0f) / 128.0f
            float[] channelData = new float[expectedOutputLength];

            // Extract only the desired channel data (don't want the alpha)
            var expectedChannelLength = expectedOutputLength / 3;
            var greenOffset = expectedChannelLength;
            var blueOffset = expectedChannelLength * 2;

            for (int i = 0, i2 = 0; i < bytes.Length; i += 4, i2++)
            {
                var r = Convert.ToSingle(bytes[i]);
                var g = Convert.ToSingle(bytes[i + 1]);
                var b = Convert.ToSingle(bytes[i + 2]);
                channelData[i2] = r;
                channelData[i2 + greenOffset] = g;
                channelData[i2 + blueOffset] = b;
            }

            return new DenseTensor<float>(new Memory<float>(channelData), new[] { 1, 3, RequiredHeight, RequiredWidth });
        }

        protected override void OnApplyObj(ObjPrediction prediction, SKPaint textPaint, SKPaint rectPaint, SKCanvas canvas)
        {
            var text = $"{prediction.Label + "|" + prediction.Confidence.ToString("0.00")}";
            var textBounds = new SKRect();
            textPaint.MeasureText(text, ref textBounds);
            canvas.DrawRect(prediction.Box.Xmin, prediction.Box.Ymin, prediction.Box.Xmax - prediction.Box.Xmin, prediction.Box.Ymax - prediction.Box.Ymin, rectPaint);
            canvas.DrawText(text, prediction.Box.Xmin, prediction.Box.Ymin - textBounds.Height, textPaint);
        }
    }
}
