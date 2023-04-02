#include "pch.h"
#include "TextEncoder.h"

using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  const size_t TextEncoder::_maxTokenCount = 77;

  TextEncoder::TextEncoder(OnnxEnvironment& environment) :
    _environment(environment),
    _sessionOptions(),
    _session(nullptr)
  {
    OrtSessionOptionsAppendExecutionProvider_DML(_sessionOptions, 0);

    _session = { _environment.Environment(), L"C:\\dev\\StableDiffusion\\StableDiffusion\\text_encoder\\model.onnx", _sessionOptions };
  }

  Tensor TextEncoder::EncodeText(Tensor text)
  {
    //Load inputs
    auto inputValue = text.ToOrtValue(_environment.MemoryInfo());

    //Bind values
    IoBinding bindings{ _session };
    bindings.BindInput("input_ids", inputValue);
    bindings.BindOutput("last_hidden_state", _environment.MemoryInfo());

    //Run inference
    _session.Run({}, bindings);

    //Get result
    auto outputValues = bindings.GetOutputValues();

    return Tensor::FromOrtValue(outputValues[0]);
  }
}