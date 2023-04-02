#pragma once
#include "pch.h"

namespace Axodox::MachineLearning
{
  struct ClipTokenizerOperator : winrt::implements<ClipTokenizerOperator, IMLOperatorKernel>
  {
    STDMETHOD(Compute)(IMLOperatorKernelContext* context) noexcept;
  };

  struct ClipTokenizerOperatorFactory : winrt::implements<ClipTokenizerOperatorFactory, IMLOperatorKernelFactory>
  {
    STDMETHOD(CreateKernel)(IMLOperatorKernelCreationContext* context, IMLOperatorKernel** kernel) noexcept;
  };

  struct ClipTokenizerShapeInferrer : winrt::implements<ClipTokenizerShapeInferrer, IMLOperatorShapeInferrer>
  {
    STDMETHOD(InferOutputShapes)(IMLOperatorShapeInferenceContext* context) noexcept;
  };

  struct ClipTokenizerTypeInferrer : winrt::implements<ClipTokenizerTypeInferrer, IMLOperatorTypeInferrer>
  {
    STDMETHOD(InferOutputTypes)(IMLOperatorTypeInferenceContext* context) noexcept;
  };

  void RegisterClipTokenizer(IMLOperatorRegistry* registry);
  void RegisterClipTokenizerSchema(IMLOperatorRegistry* registry);
}