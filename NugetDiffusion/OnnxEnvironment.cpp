#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Ort;

namespace Axodox::MachineLearning
{
  OnnxEnvironment::OnnxEnvironment() :
    _environment(),
    _memoryInfo(MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
  { }

  Ort::Env& OnnxEnvironment::Environment()
  {
    return _environment;
  }

  Ort::MemoryInfo& OnnxEnvironment::MemoryInfo()
  {
    return _memoryInfo;
  }
}