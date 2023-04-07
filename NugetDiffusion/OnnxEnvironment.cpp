#include "pch.h"
#include "OnnxEnvironment.h"

using namespace Ort;

namespace Axodox::MachineLearning
{
  OnnxEnvironment::OnnxEnvironment(const std::filesystem::path& rootPath) :
    _rootPath(rootPath),
    _environment(),
    _memoryInfo(MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault))
  {
    _defaultSessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    OrtSessionOptionsAppendExecutionProvider_DML(_defaultSessionOptions, 0);
  }

  const std::filesystem::path& OnnxEnvironment::RootPath() const
  {
    return _rootPath;
  }

  Ort::Env& OnnxEnvironment::Environment()
  {
    return _environment;
  }

  Ort::MemoryInfo& OnnxEnvironment::MemoryInfo()
  {
    return _memoryInfo;
  }

  Ort::SessionOptions& OnnxEnvironment::DefaultSessionOptions()
  {
    return _defaultSessionOptions;
  }
}