﻿using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;

namespace StableDiffusion
{
    public static class VaeDecoder
    {
        public static Tensor<float> Decoder(List<NamedOnnxValue> input)
        {
            var vaeDecoderModelPath = Directory.GetCurrentDirectory().ToString() + ("\\vae_decoder\\model.onnx");

            // Set DML EP
            SessionOptions sessionOptions = new SessionOptions();
            //sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_DML();
            // Create an InferenceSession from the Model Path.
            var vaeDecodeSession = new InferenceSession(vaeDecoderModelPath, sessionOptions);

           // Run session and send the input data in to get inference output. 
            var output = vaeDecodeSession.Run(input);
            var result = (output.ToList().First().Value as Tensor<float>);

            return result;
        }

        // create method to convert float array to an image with imagesharp
        public static Image<Rgba32> ConvertToImage(Tensor<float> output, int width = 512, int height = 512, string imageName = "sample")
        {
            var result = new Image<Rgba32>(width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }
            result.Save($@"C:/dev/{imageName}.png");

            return result;
        }
    }
}
