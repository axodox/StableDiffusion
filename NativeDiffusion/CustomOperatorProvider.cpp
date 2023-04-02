#include "pch.h"
#include "CustomOperatorProvider.h"

using namespace winrt;

namespace Axodox::MachineLearning
{
  CustomOperatorProvider::CustomOperatorProvider()
  {
    check_hresult(MLCreateOperatorRegistry(_registry.put()));
  }

  STDMETHODIMP CustomOperatorProvider::GetRegistry(IMLOperatorRegistry** operatorRegistry)
  {
    _registry.copy_to(operatorRegistry);
    return S_OK;
  }
}