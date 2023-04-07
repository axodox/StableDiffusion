#include "pch.h"
#include "StableDiffustionInferer.h"
#include "VaeDecoder.h"

using namespace DirectX;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  StableDiffusionInferer::StableDiffusionInferer(OnnxEnvironment& environment) :
    _environment(environment),
    _sessionOptions(),
    _session(nullptr),
    _floatDistribution(0.f, 1.f)
  {
    OrtSessionOptionsAppendExecutionProvider_DML(_sessionOptions, 0);
    _sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    _session = { _environment.Environment(), L"C:\\dev\\StableDiffusion\\StableDiffusion\\unet\\model.onnx", _sessionOptions };  
  }

  Tensor StableDiffusionInferer::RunInference(const StableDiffusionOptions& options)
  {
    StableDiffusionContext context{
      .Options = options,
      .Random = minstd_rand{options.Seed}
    };

    list<Tensor> derivatives;

    auto latentSample = GenerateLatentSample(context);
    auto textEmbeddings = options.TextEmbeddings.ToOrtValue(_environment.MemoryInfo());

    auto steps = context.Scheduler.GetSteps(options.StepCount);
    for (size_t i = 0; i < steps.Timesteps.size(); i++)
    {
      printf("Step %d\n", i);
      auto scaledSample = latentSample.Duplicate() / sqrt(steps.Sigmas[i] * steps.Sigmas[i] + 1);

      IoBinding binding{ _session };
      binding.BindInput("encoder_hidden_states", textEmbeddings);
      binding.BindInput("sample", scaledSample.ToOrtValue(_environment.MemoryInfo()));
      binding.BindInput("timestep", Tensor(int64_t(steps.Timesteps[i])).ToOrtValue(_environment.MemoryInfo()));
      binding.BindOutput("out_sample", _environment.MemoryInfo());
      
      _session.Run({}, binding);

      auto outputs = binding.GetOutputValues();
      auto output = Tensor::FromOrtValue(outputs[0]);
      auto outputComponents = output.Split();

      auto& blankNoise = outputComponents[0];
      auto& textNoise = outputComponents[1];
      auto guidedNoise = blankNoise.BinaryOperation<float>(textNoise, [guidanceScale = options.GuidanceScale](float a, float b) 
        { return a + guidanceScale * (b - a); });

      latentSample = steps.ApplyStep(latentSample, guidedNoise, derivatives, i);
    }

    latentSample = latentSample * (1.0f / 0.18215f);
    return latentSample;
  }
  
  Tensor StableDiffusionInferer::GenerateLatentSample(StableDiffusionContext& context)
  {
    Tensor result{ TensorType::Single, context.Options.BatchSize, 4, context.Options.Height / 8, context.Options.Width / 8 };
    
    auto initialNoiseSigma = context.Scheduler.InitialNoiseSigma();
    for (auto& value : result.AsSpan<float>())
    {
      auto u1 = _floatDistribution(context.Random);
      auto u2 = _floatDistribution(context.Random);
      auto radius = sqrt(-2.f * log(u1));
      auto theta = 2.f * XM_PI * u2;
      auto standardNormalRand = radius * cos(theta);

      value = standardNormalRand * initialNoiseSigma;
    }

    return result;
  }
}