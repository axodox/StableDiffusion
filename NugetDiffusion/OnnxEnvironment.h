#pragma once
#include "pch.h"

namespace Axodox::MachineLearning
{
  class OnnxEnvironment
  {
  public:
    OnnxEnvironment();

    Ort::Env& Environment();
    Ort::MemoryInfo& MemoryInfo();

  private:
    Ort::Env _environment;
    Ort::MemoryInfo _memoryInfo;
  };
}