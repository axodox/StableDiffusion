#include "pch.h"
#include "TextTokenizer.h"
#include "TextEncoder.h"

using namespace Axodox::MachineLearning;
using namespace std;
using namespace winrt;

int main()
{
  init_apartment();
  
  OnnxEnvironment onnxEnvironment{};
  
  //Encode text
  TextTokenizer textTokenizer{ onnxEnvironment };
  TextEncoder textEncoder{ onnxEnvironment };
    
  auto tokenizedBlank = textTokenizer.GetUnconditionalTokens();
  auto encodedBlank = textEncoder.EncodeText(tokenizedBlank);

  auto tokenizedText = textTokenizer.TokenizeText("a fireplace in an old cabin in the woods");
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

  //
  printf("done.");
}
