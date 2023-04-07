#pragma once
#include "OnnxEnvironment.h"
#include "Tensor.h"

namespace Axodox::MachineLearning
{
  class VaeDecoder
  {
  public:
    VaeDecoder(OnnxEnvironment& environment);

    Tensor DecodeVae(Tensor text);

  private:
    OnnxEnvironment& _environment;
    Ort::SessionOptions _sessionOptions;
    Ort::Session _session;
  };
}