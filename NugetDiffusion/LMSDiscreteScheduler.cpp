#include "pch.h"
#include "LMSDiscreteScheduler.h"

using namespace std;

namespace Axodox::MachineLearning
{
  LmsDiscreteScheduler::LmsDiscreteScheduler(const LmsDiscreteSchedulerOptions& options) :
    _options(options)
  {
    auto betas = _options.BetasTrained;

    if (betas.empty())
    {
      switch (_options.BetaSchedulerType)
      {
      case LmsBetaSchedulerKind::Linear:
        betas = GetLinearBetas();
        break;
      case LmsBetaSchedulerKind::ScaledLinear:
        betas = GetScaledLinearBetas();
        break;
      default:
        throw logic_error("LmsBetaSchedulerKind not implemented.");
      }
    }
    else
    {
      if (betas.size() != _options.TrainStepCount) throw invalid_argument("options.BetasTrained.Size() != options.TrainStepCount");
    }

    _cumulativeAlphas = CalculateCumulativeAlphas(betas);
    _initialNoiseSigma = CalculateInitialNoiseSigma(_cumulativeAlphas);
  }

  LmsDiscreteSchedulerSteps LmsDiscreteScheduler::GetSteps(size_t count) const
  {
    //Calculate timesteps
    vector<int32_t> timesteps;
    timesteps.resize(count);

    auto step = (_options.TrainStepCount - 1) / float(count - 1);
    for (auto value = 0.f; auto & timestep : timesteps)
    {
      timestep = int32_t(value);
      value += step;
    }

    //Calculate sigmas
    vector<float> sigmas{ _cumulativeAlphas.rbegin(), _cumulativeAlphas.rend() };
    for (auto& sigma : sigmas)
    {
      sigma = sqrt(1.f - sigma) / sigma;
    }

    vector<float> interpolatedSigmas;
    interpolatedSigmas.reserve(count);
    interpolatedSigmas.resize(count);
    for (size_t i = 0; auto & interpolatedSigma : interpolatedSigmas)
    {
      auto trainstep = timesteps[i++];
      auto previousIndex = max(size_t(floor(trainstep)), size_t(0));
      auto nextIndex = min(size_t(ceil(trainstep)), sigmas.size() - 1);
      interpolatedSigma = lerp(sigmas[previousIndex], sigmas[nextIndex], trainstep - floor(trainstep));
    }
    interpolatedSigmas.push_back(0.f);

    //Return result
    LmsDiscreteSchedulerSteps result;
    result.Timesteps = move(timesteps);
    result.Sigmas = move(interpolatedSigmas);
    return result;
  }

  float LmsDiscreteScheduler::InitialNoiseSigma() const
  {
    return _initialNoiseSigma;
  }

  std::vector<float> LmsDiscreteScheduler::GetLinearBetas() const
  {
    vector<float> results;
    results.resize(_options.TrainStepCount);

    auto value = _options.BetaAtStart;
    auto step = (_options.BetaAtEnd - _options.BetaAtStart) / (_options.TrainStepCount - 1.f);
    for (auto& beta : results)
    {
      beta = value;
      value += step;
    }

    return results;
  }

  std::vector<float> LmsDiscreteScheduler::GetScaledLinearBetas() const
  {
    vector<float> results;
    results.resize(_options.TrainStepCount);

    auto value = sqrt(_options.BetaAtStart);
    auto step = (sqrt(_options.BetaAtEnd) - value) / (_options.TrainStepCount - 1.f);
    for (auto& beta : results)
    {
      beta = value * value;
      value += step;
    }

    return results;
  }
  
  std::vector<float> LmsDiscreteScheduler::CalculateCumulativeAlphas(std::span<const float> betas)
  {
    vector<float> results{ betas.begin(), betas.end() };

    float value = 1.f;
    for (auto& result : results)
    {
      value *= 1.f - result;
      result = value;
    }

    return results;
  }
  
  float LmsDiscreteScheduler::CalculateInitialNoiseSigma(std::span<const float> cumulativeAlphas)
  {
    float result = 0;

    for (auto cumulativeAlpha : cumulativeAlphas)
    {
      auto sigma = sqrt((1.f - cumulativeAlpha) / cumulativeAlpha);
      if (sigma > result) result = sigma;
    }

    return result;
  }
}