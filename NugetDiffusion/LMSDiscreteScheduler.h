#pragma once
#include "pch.h"

namespace Axodox::MachineLearning
{
  enum class LmsBetaSchedulerKind
  {
    ScaledLinear,
    Linear
  };

  enum class LmsPredictorKind
  {
    Epsilon,
    //VPrediction
  };

  struct LmsDiscreteSchedulerOptions
  {
    size_t TrainStepCount = 1000;
    float BetaAtStart = 0.00085f;
    float BetaAtEnd = 0.012f;
    LmsBetaSchedulerKind BetaSchedulerType = LmsBetaSchedulerKind::ScaledLinear;
    LmsPredictorKind PredictorType = LmsPredictorKind::Epsilon;
    std::vector<float> BetasTrained;
  };

  struct LmsDiscreteSchedulerSteps
  {
    std::vector<int32_t> Timesteps;
    std::vector<float> Sigmas;
  };

  class LmsDiscreteScheduler
  {
  public:
    LmsDiscreteScheduler(const LmsDiscreteSchedulerOptions& options = {});

    LmsDiscreteSchedulerSteps GetSteps(size_t count) const;

    float InitialNoiseSigma() const;

    void Step(size_t step);

  private:
    LmsDiscreteSchedulerOptions _options;
    std::vector<float> _cumulativeAlphas;
    float _initialNoiseSigma;

    std::vector<float> GetLinearBetas() const;
    std::vector<float> GetScaledLinearBetas() const;
    static std::vector<float> CalculateCumulativeAlphas(std::span<const float> betas);
    static float CalculateInitialNoiseSigma(std::span<const float> cumulativeAlphas);
  };
}