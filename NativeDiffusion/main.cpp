#include "pch.h"
#include "ClipTokenizer.h"
#include "CustomOperatorProvider.h"

using namespace Axodox::MachineLearning;
using namespace std;
using namespace winrt;
using namespace Windows::Foundation;
using namespace Windows::AI::MachineLearning;

TensorInt32Bit CreateTextTensor(const wstring_view text)
{
  vector<int32_t> buffer;
  buffer.reserve(text.size());
  for (auto character : text)
  {
    buffer.push_back(character);
  }

  return TensorInt32Bit::CreateFromShapeArrayAndDataArray({ 1, int32_t(buffer.size()) }, buffer);
}

int main()
{
  init_apartment();
  Uri uri(L"http://aka.ms/cppwinrt");

  auto operatorProvider = make_self<CustomOperatorProvider>();

  com_ptr<IMLOperatorRegistry> operatorRegistry;
  check_hresult(operatorProvider->GetRegistry(operatorRegistry.put()));

  RegisterClipTokenizerSchema(operatorRegistry.get());
  RegisterClipTokenizer(operatorRegistry.get());

  auto opProvider = operatorProvider.as<ILearningModelOperatorProvider>();

  LearningModelDevice device{ LearningModelDeviceKind::Cpu };
  LearningModel model = LearningModel::LoadFromFilePath(L"C:\\dev\\StableDiffusion\\StableDiffusion\\text_tokenizer\\custom_op_cliptok.onnx", opProvider);
  
  LearningModelSession session{ model, device };
  LearningModelBinding binding{ session };

  for (auto feature : model.InputFeatures())
  {
    auto name = feature.Name();
    auto kind = feature.Kind();
    auto tensor = feature.as<TensorFeatureDescriptor>();
    auto tensorKind = tensor.TensorKind();
    auto shape = tensor.Shape();
    vector<int64_t> dimensions{ begin(shape), end(shape) };

    wprintf(L"%s\n", name.c_str());
  }

  auto inputTensor = CreateTextTensor(L"a fireplace in an old cabin in the woods");
  binding.Bind(L"input_ids", inputTensor);

  auto result = session.Evaluate(binding, L"test");
  for (auto [key, output] : result.Outputs())
  {
    auto result = output.as<TensorFloat>();
    auto shape = result.Shape();
    vector<int64_t> dimensions{ begin(shape), end(shape) };

    wprintf(L"%s\n", key.c_str());
  }

  printf("Hello, %ls!\n", uri.AbsoluteUri().c_str());
}