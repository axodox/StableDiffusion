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

    _session = { _environment.Environment(), L"C:\\dev\\StableDiffusion\\StableDiffusion\\unet\\model.onnx", _sessionOptions };  
  }

  float ClosedIntegral(const std::function<float(float)>& f, float intervalStart, float intervalEnd, float targetError = 1e-4) 
  {
    float h = intervalEnd - intervalStart;  // step size
    float sum = 0.5f * (f(intervalStart) + f(intervalEnd));  // initial sum

    int iterationCount = 1;  // number of iterations
    float actualError = targetError + 1.f;  // initialize actual error

    while (actualError > targetError) 
    {
      float x = intervalStart + 0.5f * h;
      float partialSum = 0.f;

      for (int i = 0; i < iterationCount; i++) {
        partialSum += f(x);
        x += h;
      }

      sum += partialSum;
      h *= 0.5f;
      iterationCount *= 2;
      actualError = abs(partialSum * h);
    }

    return sum * h;
  }

  Tensor StableDiffusionInferer::RunInference(const StableDiffusionOptions& options)
  {
    StableDiffusionContext context{
      .Options = options,
      .Random = minstd_rand{options.Seed}
    };

    //
    list<Tensor> derivatives;
    auto order = 4;
    //


    auto latentSample = GenerateLatentSample(context).Duplicate();
    auto textEmbeddings = options.TextEmbeddings.ToOrtValue(_environment.MemoryInfo());

    auto steps = context.Scheduler.GetSteps(options.StepCount);
    for (size_t i = 0; i < steps.Timesteps.size(); i++)
    {
      auto scaledSample = latentSample / sqrt(steps.Sigmas[i] * steps.Sigmas[i] + 1);

      IoBinding binding{ _session };
      binding.BindInput("encoder_hidden_states", textEmbeddings);
      binding.BindInput("sample", scaledSample.ToOrtValue(_environment.MemoryInfo()));
      binding.BindInput("timestep", Tensor(int64_t(steps.Timesteps[i])).ToOrtValue(_environment.MemoryInfo()));
      
      _session.Run({}, binding);

      auto outputs = binding.GetOutputValues();
      auto output = Tensor::FromOrtValue(outputs[0]);
      auto outputComponents = output.Split();

      auto& blankNoise = outputComponents[0];
      auto& textNoise = outputComponents[1];
      auto guidedNoise = blankNoise.BinaryOperation<float>(textNoise, [guidanceScale = options.GuidanceScale](float a, float b) 
        { return a + guidanceScale * (b - a); });

      //
      auto stepIndex = i;
      auto sigma = steps.Sigmas[i];

      auto& modelOutput = guidedNoise;
      auto& sample = latentSample;

      // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
      Tensor predictedOriginalSample;

      //if(predictionType == "epsilon")
      {
        predictedOriginalSample = sample.BinaryOperation<float>(modelOutput, [sigma](float a, float b) { return a - sigma * b; });
      }

      // 2. Convert to an ODE derivative
      Tensor derivativeItemsArray = sample.BinaryOperation<float>(predictedOriginalSample, [sigma](float a, float b) { return (a - b) / sigma; });

      derivatives.push_back(derivativeItemsArray);
      if (derivatives.size() > order) derivatives.pop_front();

      // 3. compute linear multistep coefficients
      auto LmsDerivative = [&steps, order, t = stepIndex](float tau, int currentOrder)
      {
        float prod = 1.f;
        for (int k = 0; k < order; k++)
        {
          if (currentOrder == k)
          {
            continue;
          }
          prod *= (tau - steps.Sigmas[t - k]) / (steps.Sigmas[t - currentOrder] - steps.Sigmas[t - k]);
        }
        return prod;
      };

      vector<float> lmsCoeffs;
      lmsCoeffs.reserve(derivatives.size());
      for (auto t = 0; t < derivatives.size(); t++)
      {
        lmsCoeffs.push_back(ClosedIntegral([&](float tau) { return LmsDerivative(tau, t); }, steps.Sigmas[t], steps.Sigmas[t+1]));
      }

      // 4. compute previous sample based on the derivative path
      // Reverse list of tensors this.derivatives
      derivatives.reverse();

      // Create list of tuples from the lmsCoeffs and reversed derivatives
      vector<pair<float, Tensor>> lmsCoeffsAndDerivatives;
      lmsCoeffsAndDerivatives.reserve(derivatives.size());
      for (auto x = 0; auto& derivative : derivatives)
      {
        lmsCoeffsAndDerivatives.push_back({ lmsCoeffs[x++], derivatives });
      }

      // Create tensor for product of lmscoeffs and derivatives
      vector<Tensor> lmsDerProduct;
      lmsDerProduct.reserve(derivatives.size());

      for (auto& [lmsCoeff, derivative] : lmsCoeffsAndDerivatives)
      {
        // Multiply to coeff by each derivatives to create the new tensors
        lmsDerProduct.push_back(derivative * lmsCoeff);
      }

      // Sum the tensors
      Tensor sumTensor{ TensorType::Single, derivativeItemsArray.Shape };
      for (auto& tensor : lmsDerProduct)
      {
        sumTensor.UnaryOperation<float>(tensor, [](float a, float b) { return a + b; });
      }

      // Add the sumed tensor to the sample
      auto prevSample = sample.BinaryOperation<float>(sumTensor, [](float a, float b) { return a + b; });

      latentSample = prevSample;
    }

    latentSample = latentSample * (1.0f / 0.18215f);
    return latentSample;
  }
  
  Tensor StableDiffusionInferer::GenerateLatentSample(StableDiffusionContext& context)
  {
    Tensor result{ TensorType::Single, context.Options.BatchSize, 4, context.Options.Width / 8, context.Options.Height / 8 };
    
    for (auto& value : result.AsSpan<float>())
    {
      auto u1 = _floatDistribution(context.Random);
      auto u2 = _floatDistribution(context.Random);
      auto radius = sqrt(-2.f * log(u1));
      auto theta = 2.f * XM_PI * u2;
      auto standardNormalRand = radius * cos(theta);

      value = standardNormalRand * context.Scheduler.InitialNoiseSigma();
    }

    return result;
  }
}