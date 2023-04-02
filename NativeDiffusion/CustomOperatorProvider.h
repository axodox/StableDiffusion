#pragma once
#include "pch.h"

namespace Axodox::MachineLearning
{
  struct CustomOperatorProvider : winrt::implements<CustomOperatorProvider,
    winrt::Windows::AI::MachineLearning::ILearningModelOperatorProvider,
    ILearningModelOperatorProviderNative>
  {
    CustomOperatorProvider();

    STDMETHOD(GetRegistry)(IMLOperatorRegistry** operatorRegistry);

  private:
    winrt::com_ptr<IMLOperatorRegistry> _registry;
  };
}