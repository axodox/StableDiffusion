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
  
  OnnxEnvironment onnxEnvironment{ L"C:\\dev\\StableDiffusion\\StableDiffusion\\" };
  
  //Encode text
  TextTokenizer textTokenizer{ onnxEnvironment };
  TextEncoder textEncoder{ onnxEnvironment };
    
  auto tokenizedBlank = textTokenizer.GetUnconditionalTokens();
  auto encodedBlank = textEncoder.EncodeText(tokenizedBlank);

  auto tokenizedText = textTokenizer.TokenizeText("beautiful landscape with a mountain and a lake");
  auto encodedText = textEncoder.EncodeText(tokenizedText);
    
  //Create text embeddings
  Tensor textEmbeddings{ TensorType::Single, 2, 77, 768 };

  auto pSourceBlank = encodedBlank.AsPointer<float>();
  auto pSourceText = encodedText.AsPointer<float>();
  auto pTargetBlank = textEmbeddings.AsPointer<float>(0);
  auto pTargetText = textEmbeddings.AsPointer<float>(1);

  auto size = encodedText.Size();
  for (size_t i = 0; i < size; i++)
  {
    *pTargetBlank++ = *pSourceBlank++;
    *pTargetText++ = *pSourceText++;
  }

  //Run stable diffusion
  StableDiffusionInferer stableDiffusion{ onnxEnvironment };

  StableDiffusionOptions options {
    .Seed = 100,
    .TextEmbeddings = textEmbeddings
  };
  Tensor latentResult = stableDiffusion.RunInference(options);

  //Decode VAE
  VaeDecoder vaeDecoder{ onnxEnvironment };
  auto imageTensor = vaeDecoder.DecodeVae(latentResult);

  auto imageTexture = imageTensor.ToTextureData();
  auto pngBuffer = imageTexture[0].ToBuffer();
  write_file(L"bin/test.png", pngBuffer);

  //Done
  printf("done.");
}
