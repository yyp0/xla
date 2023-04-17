#include "torch_xla/csrc/xla_auto_sharding.h"

#include "tensorflow/compiler/xla/hlo/experimental/auto_sharding/auto_sharding.h"
#include "tensorflow/compiler/xla/hlo/transforms/hlo_constant_splitter.h"
#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/all_reduce_reassociate.h"
#include "tensorflow/compiler/xla/service/call_inliner.h"
#include "tensorflow/compiler/xla/service/conditional_canonicalizer.h"
#include "tensorflow/compiler/xla/service/conditional_simplifier.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/dot_merger.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/gather_simplifier.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_shape_verifier.h"
// #include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/scatter_expander.h"
#include "tensorflow/compiler/xla/service/scatter_simplifier.h"
#include "tensorflow/compiler/xla/service/sharding_propagation.h"
#include "tensorflow/compiler/xla/service/sharding_remover.h"
#include "tensorflow/compiler/xla/service/sort_simplifier.h"
#include "tensorflow/compiler/xla/service/spmd/collective_permute_motion.h"
#include "tensorflow/compiler/xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "tensorflow/compiler/xla/service/transpose_folding.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/while_loop_constant_sinking.h"
#include "tensorflow/compiler/xla/service/while_loop_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "third_party/xla_client/debug_macros.h"
#include "third_party/xla_client/tf_logging.h"
#include "third_party/xla_client/xla_util.h"

namespace torch_xla {

xla::HloModuleProto AutoShardingRunner::Run(
    const xla::XlaComputation& computation) {
  // transform computation to xla module
  TF_VLOG(6) << "\n\n------------> start transform computation to module ";
  auto hlo_text_error = xla::util::GetComputationHloText(computation);
  XLA_CHECK_OK(hlo_text_error.status())
      << "Converts a computation to textual HLO form failed: "
      << hlo_text_error.status();

  int64_t num_replicas = 1;
  int64_t num_devices = 8;

  xla::HloModuleConfig config(computation.GetProgramShape().value());
  config.set_static_device_assignment(xla::DeviceAssignment(1, num_devices));
  config.set_use_spmd_partitioning(true);
  config.set_replica_count(num_replicas);
  config.set_num_partitions(num_devices);
  config.set_allow_spmd_sharding_propagation_to_output(
      absl::MakeConstSpan({true}));

  auto hlo_module_error =
      xla::ParseAndReturnUnverifiedModule(hlo_text_error.value(), config);
  XLA_CHECK_OK(hlo_module_error.status())
      << "HLO Module loading failed: " << hlo_module_error.status();
  auto module = std::move(hlo_module_error.value());
  TF_VLOG(6) << "------------> module before sharding pass ";
  TF_VLOG(6) << module->ToString();
  return Run(module.get());
}

xla::HloModuleProto AutoShardingRunner::Run(xla::HloModule* module) {
  TF_VLOG(6) << "\n\n------------> start auto sharding pass";
  xla::AutoShardingOption option;
  option.enable = true;
  option.device_mesh_shape = {1, 4};
  option.device_mesh_ids = {0, 1, 2, 3};
  option.device_mesh_alpha = {1.0, 1.0};
  option.device_mesh_beta = {0.01, 1.0};

  xla::HloPassPipeline spmd_pipeline("run-auto-sharding");
  spmd_pipeline.AddPass<xla::CallInliner>();
  spmd_pipeline.AddPass<xla::DotDecomposer>();
  spmd_pipeline.AddPass<xla::ZeroSizedHloElimination>();
  spmd_pipeline.AddPass<xla::ConditionalCanonicalizer>();

  xla::HloPassPipeline& spmd_simplify =
      spmd_pipeline.AddPass<xla::HloPassFix<xla::HloPassPipeline>>(
          "spmd-simplify");

  //   spmd_simplify.AddPass<xla::AlgebraicSimplifier>(
  //       layout_insensitive_algsimp_opts);
  spmd_simplify.AddPass<xla::SortSimplifier>();
  spmd_simplify.AddPass<xla::TupleSimplifier>();
  // spmd_simplify.AddPass<ScatterSimplifier>();
  spmd_simplify.AddPass<xla::ScatterExpander>(
      xla::ScatterExpander::kEliminateSimpleScatters);
  // spmd_simplify.AddPass<GatherSimplifier>();
  spmd_simplify.AddPass<xla::GatherExpander>(
      xla::GatherExpander::kEliminateSimpleGathers);
  spmd_simplify.AddPass<xla::WhileLoopConstantSinking>();
  spmd_simplify.AddPass<xla::WhileLoopSimplifier>();

  spmd_simplify.AddPass<xla::ReshapeMover>();
  spmd_simplify.AddPass<xla::HloConstantFolding>();
  spmd_simplify.AddPass<xla::ConditionalSimplifier>();
  //   spmd_simplify.AddPass<xla::TransposeFolding>(
  //       xla::gpu::CanFoldTransposeOperandIntoDot);
  spmd_simplify.AddPass<xla::HloCSE>(
      /*is_layout_sensitive=*/false);
  spmd_simplify.AddPass<xla::HloDCE>();

  spmd_pipeline.AddPass<xla::HloConstantSplitter>();

  spmd_pipeline.AddPass<xla::AutoSharding>(option);
  spmd_pipeline.AddPass<xla::ShardingPropagation>(
      /*is_spmd=*/true, /*propagate_metadata=*/false,
      /*allow_spmd_sharding_propagation_to_output=*/
      absl::MakeConstSpan({true}));
  //   spmd_pipeline.AddPass<xla::spmd::StatefulRngSpmdPartitioner>(
  //       4, module->config().replica_count());
  //   spmd_pipeline.AddPass<xla::CollectivePermuteMotion>();
  // spmd_pipeline.AddPass<SliceAutoShardedStages>();

  const auto& pass_status = spmd_pipeline.Run(module).status();
  if (!pass_status.ok()) {
    XLA_ERROR() << "spmd-partitioning pass failed";
  }
  TF_VLOG(6) << "------------> module after auto sharding pass";
  TF_VLOG(6) << module->ToString();
  //   return module->ToProto();

  xla::HloPassPipeline pipe("run-spmd-partitioner");
  pipe.AddPass<xla::ShardingPropagation>(
      /*is_spmd=*/true, /*propagate_metadata=*/false,
      /*allow_spmd_sharding_propagation_to_output=*/
      absl::MakeConstSpan({true}));
  pipe.AddPass<xla::spmd::StatefulRngSpmdPartitioner>(
      4, module->config().replica_count());
  pipe.AddPass<xla::CollectivePermuteMotion>();
  pipe.AddPass<xla::AllReduceReassociate>();
  const auto& pip_status = pipe.Run(module).status();
  if (!pip_status.ok()) {
    XLA_ERROR() << "run spmd partitioner pass failed.";
  }

  TF_VLOG(6) << "------------> module after spmd partitioning pass";
  TF_VLOG(6) << module->ToString();

  return module->ToProto();
}

}  // namespace torch_xla