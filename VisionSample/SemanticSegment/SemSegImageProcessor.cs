using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;
using System;
using System.Collections.Generic;
using System.Text;

namespace VisionSample
{
    public class SemSegImageProcessor : SkiaSharpImageProcessor<SKBitmap, float>
    {
        const int RequiredWidth = 2048;
        const int RequiredHeight = 1024;

        protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
           => sourceImage.Resize(new SKImageInfo(RequiredWidth, RequiredHeight), SKFilterQuality.None);

        protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
        {
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 3, RequiredHeight, RequiredWidth });

            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    input[0, 0, y, x] = ((pixel.Red / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixel.Green / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixel.Blue / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;
        }
    }
}
