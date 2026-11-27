# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Legacy class paths for loading checkpoints saved before the PhysFlowNet release."""

from .physflownet import (
    AdaGN,
    DiagonalSPDMetricEstimator,
    HPCE as PhysicsContext,
    PMD as PhysicalManifoldDown,
    PPRB as C3k2_Patch_SB,
    PhysFlowNetDetect as YORNDetect,
    PhysFlowNetNeckRefinement as YORN_Manifold_Wrapper,
    PhysicsRiemannianFlowBlock,
)

FeatureRefiner = YORN_Manifold_Wrapper
