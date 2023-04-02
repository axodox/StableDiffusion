#include "pch.h"
#include "ClipTokenizer.h"

using namespace std;
using namespace winrt;

namespace Axodox::MachineLearning
{
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  const char* _domain = "ai.onnx.contrib";
  const char* _name = "CLIPTokenizer";

  HRESULT GetTensorMutableDataString(IMLOperatorTensor* tensor, std::vector<std::string>& output)
  {
    HRESULT result = S_OK;

    //Get text length
    /*vector<uint32_t> dimensions;
    dimensions.resize(tensor->GetDimensionCount());
    result = tensor->GetShape(uint32_t(dimensions.size()), dimensions.data());
    if (FAILED(result)) return result;*/

    //tensor->()

    return result;
  }

  STDMETHODIMP ClipTokenizerOperator::Compute(IMLOperatorKernelContext* context) noexcept
  {
    HRESULT result = S_OK;

    com_ptr<IMLOperatorTensor> input;
    result = context->GetInputTensor(0, input.put());
    if (FAILED(result)) return result;

    /*std::vector<std::string> str_input;
    std::list<OffsetMappingType> offset_map;*/

    return result;
  }

  STDMETHODIMP ClipTokenizerOperatorFactory::CreateKernel(IMLOperatorKernelCreationContext* context, IMLOperatorKernel** kernel) noexcept
  {
    auto clipTokenizer = make<ClipTokenizerOperator>();
    clipTokenizer.copy_to(kernel);
    return S_OK;
  }

  STDMETHODIMP ClipTokenizerShapeInferrer::InferOutputShapes(IMLOperatorShapeInferenceContext* context) noexcept
  {
    return S_OK;
  }

  STDMETHODIMP ClipTokenizerTypeInferrer::InferOutputTypes(IMLOperatorTypeInferenceContext* context) noexcept
  {
    MLOperatorEdgeDescription int64TypeDescription{
      .edgeType = MLOperatorEdgeType::Tensor,
      .tensorDataType = MLOperatorTensorDataType::Int64
    };

    context->SetOutputEdgeDescription(0, &int64TypeDescription);
    context->SetOutputEdgeDescription(1, &int64TypeDescription);

    return S_OK;
  }

  void RegisterClipTokenizer(IMLOperatorRegistry* registry)
  {
    MLOperatorEdgeDescription stringTypeDescription{
      .edgeType = MLOperatorEdgeType::Tensor,
      .tensorDataType = MLOperatorTensorDataType::String
    };

    MLOperatorEdgeTypeConstraint stringConstraint{
      .typeLabel = "string_t",
      .allowedTypes = &stringTypeDescription,
      .allowedTypeCount = 1
    };

    MLOperatorEdgeDescription int64TypeDescription{
      .edgeType = MLOperatorEdgeType::Tensor,
      .tensorDataType = MLOperatorTensorDataType::Int64
    };

    MLOperatorEdgeTypeConstraint int64Constraint{
      .typeLabel = "int64_t",
      .allowedTypes = &int64TypeDescription,
      .allowedTypeCount = 1
    };

    MLOperatorEdgeTypeConstraint typeConstraints[] = {
      stringConstraint,
      int64Constraint
    };

    MLOperatorKernelDescription kernelDescription{
      .domain = _domain,
      .name = _name,
      .minimumOperatorSetVersion = 7,
      .executionType = MLOperatorExecutionType::Cpu,
      .typeConstraints = typeConstraints,
      .typeConstraintCount = size(typeConstraints),
      .defaultAttributes = nullptr,
      .defaultAttributeCount = 0,
      .options = MLOperatorKernelOptions::None,
      .executionOptions = 0
    };

    static auto factory = make<ClipTokenizerOperatorFactory>();
    static auto shapeInferrer = make<ClipTokenizerShapeInferrer>();
    check_hresult(registry->RegisterOperatorKernel(&kernelDescription, factory.get(), shapeInferrer.get()));
  }

  void RegisterClipTokenizerSchema(IMLOperatorRegistry* registry)
  {
    MLOperatorSetId operatorSetId{
      .domain = _domain,
      .version = 7
    };

    MLOperatorSchemaEdgeDescription inputDescription{
      .options = MLOperatorParameterOptions::Single,
      .typeFormat = MLOperatorSchemaEdgeTypeFormat::Label,
      .typeLabel = "string_t"
    };

    MLOperatorSchemaEdgeDescription outputDescription{
      .options = MLOperatorParameterOptions::Single,
      .typeFormat = MLOperatorSchemaEdgeTypeFormat::Label,
      .typeLabel = "int64_t"
    };

    MLOperatorSchemaEdgeDescription outputs[] = {
      outputDescription,
      outputDescription
    };

    MLOperatorEdgeDescription stringTypeDescription{
      .edgeType = MLOperatorEdgeType::Tensor,
      .tensorDataType = MLOperatorTensorDataType::String
    };

    MLOperatorEdgeTypeConstraint stringConstraint{
      .typeLabel = "string_t",
      .allowedTypes = &stringTypeDescription,
      .allowedTypeCount = 1
    };

    MLOperatorEdgeDescription int64TypeDescription{
      .edgeType = MLOperatorEdgeType::Tensor,
      .tensorDataType = MLOperatorTensorDataType::Int64
    };

    MLOperatorEdgeTypeConstraint int64Constraint{
      .typeLabel = "int64_t",
      .allowedTypes = &int64TypeDescription,
      .allowedTypeCount = 1
    };

    MLOperatorEdgeTypeConstraint typeConstraints[] = {
      stringConstraint,
      int64Constraint
    };

    MLOperatorAttribute attributes[] = {
      { "merges", MLOperatorAttributeType::String, 1 },
      { "padding_length", MLOperatorAttributeType::Int, 1 },
      { "vocab", MLOperatorAttributeType::String, 1 }
    };    

    MLOperatorSchemaDescription operatorSchemaDescription{
      .name = _name,
      .operatorSetVersionAtLastChange = 1,
      .inputs = &inputDescription,
      .inputCount = 1,
      .outputs = outputs,
      .outputCount = size(outputs),
      .typeConstraints = typeConstraints,
      .typeConstraintCount = size(typeConstraints),
      .attributes = attributes,
      .attributeCount = size(attributes),
      .defaultAttributes = nullptr,
      .defaultAttributeCount = 0
    };

    static auto shapeInferrer = make<ClipTokenizerShapeInferrer>();
    static auto typeInferrer = make<ClipTokenizerTypeInferrer>();

    auto schemas = &operatorSchemaDescription;
    check_hresult(registry->RegisterOperatorSetSchema(&operatorSetId, 7, &schemas, 1, typeInferrer.get(), shapeInferrer.get()));
  }
}