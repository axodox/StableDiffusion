#include "pch.h"
#include "TextTokenizer.h"
#include "TextEncoder.h"
#include "StableDiffustionInferer.h"
#include "VaeDecoder.h"
#include "Storage/FileIO.h"

using namespace Axodox::MachineLearning;
using namespace Axodox::Storage;
using namespace std;
using namespace winrt;

int main()
{
  init_apartment();
  
  OnnxEnvironment onnxEnvironment{ L"C:/dev/ai/realistic_vision_v1.4-fp16-vram" };

  //Create text embeddings
  Tensor textEmbeddings{ TensorType::Single, 2, 77, 768 };
  {
    //Encode text
    TextTokenizer textTokenizer{ onnxEnvironment };
    TextEncoder textEncoder{ onnxEnvironment };

    auto tokenizedBlank = textTokenizer.GetUnconditionalTokens();
    auto encodedBlank = textEncoder.EncodeText(tokenizedBlank);

    auto tokenizedText = textTokenizer.TokenizeText("a stag standing in a misty forest at dawn");
    auto encodedText = textEncoder.EncodeText(tokenizedText);

    auto pSourceBlank = encodedBlank.AsPointer<float>();
    auto pSourceText = encodedText.AsPointer<float>();
    auto pTargetBlank = textEmbeddings.AsPointer<float>(0);
    auto pTargetText = textEmbeddings.AsPointer<float>(1);

    auto size = encodedText.Size();
    memcpy(pTargetBlank, pSourceBlank, size * 4);
    memcpy(pTargetText, pSourceText, size * 4);
  }

  //Run stable diffusion
  Tensor latentResult;
  {
    StableDiffusionInferer stableDiffusion{ onnxEnvironment };

    StableDiffusionOptions options{
      .StepCount = 15,
      .Width = 768,
      .Height = 768,
      .Seed = 50,
      .TextEmbeddings = textEmbeddings
    };
    
    latentResult = stableDiffusion.RunInference(options);
  }

  //Decode VAE
  {
    VaeDecoder vaeDecoder{ onnxEnvironment };
    auto imageTensor = vaeDecoder.DecodeVae(latentResult);

    auto imageTexture = imageTensor.ToTextureData();
    auto pngBuffer = imageTexture[0].ToBuffer();
    write_file(L"bin/test.png", pngBuffer);
  }

  //Done
  printf("done.");
}
