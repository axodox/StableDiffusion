#include "pch.h"
#include "TextTokenizer.h"
#include "OnnxExtensions.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  const size_t TextTokenizer::_maxTokenCount = 77;
  const int32_t TextTokenizer::_blankToken = 49407;

  TextTokenizer::TextTokenizer(OnnxEnvironment& environment) :
    _environment(environment),
    _sessionOptions(),
    _session(nullptr)
  {
    _sessionOptions.RegisterCustomOpsLibrary(L"C:\\dev\\StableDiffusion\\StableDiffusion\\ortextensions.dll");
    _session = { _environment.Environment(), L"C:\\dev\\StableDiffusion\\StableDiffusion\\text_tokenizer\\custom_op_cliptok.onnx", _sessionOptions };
  }

  Tensor TextTokenizer::TokenizeText(const std::string_view text)
  {
    //Load inputs
    Allocator allocator{ _session, _environment.MemoryInfo() };
    
    vector<int64_t> inputShape{ 1 };
    auto inputValue = Value::CreateTensor(allocator, inputShape.data(), inputShape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
    
    const char* inputLines[] = { text.data() };
    inputValue.FillStringTensor(inputLines, size(inputLines));

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("string_input", inputValue);
    bindings.BindOutput("input_ids", _environment.MemoryInfo());
    bindings.BindOutput("attention_mask", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();
    auto outputSpan = AsSpan<int64_t>(outputValues[0]);

    //Pad results to a fixed size
    Tensor result{ TensorType::Int32, 1, _maxTokenCount };

    auto sToken = result.At<int32_t>();
    auto pToken = sToken;
    for (auto token : outputSpan)
    {
      *pToken++= int32_t(token);
    }

    memset(pToken, _blankToken, pToken - sToken);

    return result;
  }
  
  Tensor TextTokenizer::GetUnconditionalTokens()
  {
    Tensor result{ TensorType::Int32, 1, _maxTokenCount };
    memset(result.Buffer.data(), _blankToken, result.Buffer.size());

    *result.At<int32_t>(0) = 49406;
    return result;
  }
}