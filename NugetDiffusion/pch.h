#pragma once
#include <span>
#include <ranges>
#include <random>
#include <functional>

#define NOMINMAX
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>

#include <DirectXMath.h>

#include "onnxruntime_cxx_api.h"
#include "dml_provider_factory.h"