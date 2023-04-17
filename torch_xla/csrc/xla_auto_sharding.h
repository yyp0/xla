#pragma once

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"

namespace torch_xla {

class AutoShardingRunner {
 public:
  xla::HloModuleProto Run(const xla::XlaComputation& computation);
  xla::HloModuleProto Run(xla::HloModule* module);
};

}  // namespace torch_xla