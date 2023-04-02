#include "pch.h"

using namespace std;
using namespace winrt;
using namespace Windows::Foundation;
using namespace Ort;

vector<int32_t> TokenizeText(const string_view text)
{ 
  Env environment{};

  SessionOptions sessionOptions{};
  sessionOptions.RegisterCustomOpsLibrary(L"C:\\dev\\StableDiffusion\\StableDiffusion\\ortextensions.dll");
   

  Session tokenizeSession{ environment, L"C:\\dev\\StableDiffusion\\StableDiffusion\\text_tokenizer\\custom_op_cliptok.onnx", sessionOptions };
  
  auto typeInfo = tokenizeSession.GetInputTypeInfo(0);
  auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
  auto elementType = tensorInfo.GetElementType();
  auto tensorShape = tensorInfo.GetShape();

  IoBinding binding{tokenizeSession};

  auto memoryInfo = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    
  Allocator allocator{ tokenizeSession, memoryInfo};
  vector<int64_t> shape{ 1 };
  //auto value = Value::CreateTensor<string>(memoryInfo, &text, 1, shape.data(), shape.size());  
  auto value = Value::CreateTensor(allocator, shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  
  const char* lines[] = { text.data() };
  value.FillStringTensor(lines, size(lines));
  
  binding.BindInput("string_input", value);
  binding.BindOutput("input_ids", memoryInfo);
  binding.BindOutput("attention_mask", memoryInfo);

  RunOptions options{};
  tokenizeSession.Run(options, binding);

  auto results = binding.GetOutputValues();
  auto x = results[0].GetTensorData<int64_t>();
  auto y = results[0].GetTensorTypeAndShapeInfo().GetShape();
  printf("asd");

  return{};
}

int main()
{
  init_apartment();
  Uri uri(L"http://aka.ms/cppwinrt");
  printf("Hello, %ls!\n", uri.AbsoluteUri().c_str());

  string text = "Some text.";
  TokenizeText(text);
}
