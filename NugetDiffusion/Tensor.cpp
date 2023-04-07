#include "pch.h"
#include "Tensor.h"

using namespace Axodox::Graphics;
using namespace DirectX;
using namespace DirectX::PackedVector;
using namespace Ort;
using namespace std;

namespace Axodox::MachineLearning
{
  Tensor::Tensor() :
    Type(TensorType::Unknown),
    Shape({ 0, 0, 0, 0 })
  { }

  Tensor::Tensor(TensorType type, size_t x, size_t y, size_t z, size_t w) :
    Type(type),
    Shape({x, y, z, w})
  {
    AllocateBuffer();
  }

  Tensor::Tensor(TensorType type, shape_t shape) :
    Type(type),
    Shape(shape)
  {
    AllocateBuffer();
  }

  void Tensor::AllocateBuffer()
  {
    Buffer.resize(ByteCount());
  }

  size_t Tensor::ByteCount() const
  {
    size_t result = Shape[0] > 0 ? GetElementSize(Type) : 0;

    for (auto dimension : Shape)
    {
      if (dimension != 0) result *= dimension;
    }

    return result;
  }

  bool Tensor::IsValid() const
  {
    return Type != TensorType::Unknown && !Buffer.empty() && Buffer.size() == ByteCount();
  }

  void Tensor::ThrowIfInvalid() const
  {
    if (IsValid()) throw runtime_error("The tensor is invalid.");
  }

  size_t Tensor::Size(size_t index) const
  {
    size_t result = 1;

    for (auto i = index; i < Shape.size(); i++)
    {
      if (Shape[i] > 0) result *= Shape[i];
    }

    return result;
  }

  Tensor Tensor::FromOrtValue(const Ort::Value& value)
  {
    auto info = value.GetTensorTypeAndShapeInfo();

    Tensor result;

    //Set shape
    auto shape = info.GetShape();
    if (shape.size() > result.Shape.size()) throw logic_error("Tensor does not support more than 4 dimensions.");

    for (auto i = 0; auto dimension : shape)
    {
      if (dimension > 0) result.Shape[i++] = size_t(dimension);
    }

    //Set type
    result.Type = ToTensorType(info.GetElementType());

    //Allocate buffer
    result.AllocateBuffer();

    //Copy data
    memcpy(result.Buffer.data(), value.GetTensorRawData(), result.Buffer.size());

    return result;
  }

  Ort::Value Tensor::ToOrtValue(Ort::MemoryInfo& memoryInfo) const
  {
    std::vector<int64_t> shape;
    for (auto dimension : Shape)
    {
      if (dimension != 0) shape.push_back(int64_t(dimension));
    }

    return Value::CreateTensor(memoryInfo, const_cast<uint8_t*>(Buffer.data()), Buffer.size(), shape.data(), shape.size(), ToTensorType(Type));
  }

  std::vector<Graphics::TextureData> Tensor::ToTextureData() const
  {
    if (Type != TensorType::Single) throw bad_cast();

    vector<TextureData> results;
    results.reserve(Shape[0]);

    auto width = uint32_t(Shape[2]);
    auto height = uint32_t(Shape[3]);
    for (size_t i = 0u; i < Shape[0]; i++)
    {
      TextureData result{ width, height, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB };
      
      auto pTarget = result.Row<XMBYTEN4>(0);
      auto rSource = AsPointer<float>(i, 0);
      auto gSource = AsPointer<float>(i, 1);
      auto bSource = AsPointer<float>(i, 2);
      for (size_t y = 0u; y < Shape[2]; y++)
      {
        for (size_t x = 0u; x < Shape[3]; x++)
        {
          *pTarget++ = XMBYTEN4{ *rSource++, *gSource++, *rSource++, 1.f };
        }
      }

      results.push_back(move(result));
    }

    return results;
  }

  const uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w) const
  {
    shape_t index{ x, y, z, w };

    auto elementSize = GetElementSize(Type);
    size_t offset = 0;
    for (size_t i = 0; i < Shape.size(); i++)
    {
      offset += index[i] * Size(i + 1) * elementSize;
    }

    if (offset > Buffer.size()) throw out_of_range("Tensor index out of range.");

    return Buffer.data() + offset;
  }

  uint8_t* Tensor::AsPointer(size_t x, size_t y, size_t z, size_t w)
  {
    return const_cast<uint8_t*>(static_cast<const Tensor*>(this)->AsPointer(x, y, z, w));
  }
  
  Tensor Tensor::Duplicate(size_t instances) const
  {
    Tensor tensor{ Type, Shape[0] * instances, Shape[1], Shape[2], Shape[3] };
    
    for (auto i = 0; i < instances; i++)
    {
      memcpy(tensor.AsPointer(i), AsPointer(), ByteCount());
    }

    return tensor;
  }
  
  std::vector<Tensor> Tensor::Split(size_t instances) const
  {
    if (Shape[0] % instances != 0) throw invalid_argument("instances");

    auto newShape = Shape;
    newShape[0] /= instances;

    vector<Tensor> results;
    results.resize(instances);
    for (size_t i = 0; auto & result : results)
    {
      result = Tensor(Type, newShape);
      memcpy(result.AsPointer(), AsPointer(i++), result.ByteCount());
    }

    return results;
  }
}