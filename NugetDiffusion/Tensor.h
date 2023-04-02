#pragma once
#include "TensorType.h"

namespace Axodox::MachineLearning
{
  struct Tensor
  {
    typedef std::array<size_t, 4> shape_t;

    TensorType Type;
    shape_t Shape;
    std::vector<uint8_t> Buffer;

    Tensor();
    Tensor(TensorType type, size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0);

    void AllocateBuffer();

    size_t ByteCount() const;
    bool IsValid() const;
    void ThrowIfInvalid() const;

    size_t Size(size_t dimension = 0) const;

    static Tensor FromOrtValue(const Ort::Value& value);
    Ort::Value ToOrtValue(Ort::MemoryInfo& memoryInfo) const;

    template<typename T>
    T* At(size_t x = 0, size_t y = 0, size_t z = 0, size_t w = 0)
    {
      if (ToTensorType<T>() != Type) throw std::bad_cast();

      shape_t index{ x, y, z, w };
      
      auto offset = sizeof(T);
      for (size_t i = 0; i < Shape.size(); i++)
      {
        offset += index[i] * Size(i + 1);
      }

      if (offset > Buffer.size()) throw std::out_of_range("Tensor index out of range.");

      return reinterpret_cast<T*>(Buffer.data() + offset);
    }
  };
}