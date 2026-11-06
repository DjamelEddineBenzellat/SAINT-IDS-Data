"""
X3 configurations of the API for the component-cost comparison (uvicorn factory, $X2_CONFIG):
  L0  LightGBM only: ONNX + decide() with the ML source only (rule detectors disabled in the policy)
  L1  + rule detectors
  L2  + SHAP explanations (native, top-5)
  L3  + CKKS-encrypted relay (= the deployed full system; needs ALERT_WEBHOOK_URL etc.)
All use the same alert policy (tau = 0.25) and the same serving model. No Isolation Forest (policy 2.2.0).
  uvicorn x2_app:create --factory
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import replace


def create():
    cfg = os.environ["X2_CONFIG"]
    if cfg != "L3":
        os.environ.pop("ALERT_WEBHOOK_URL", None)          # no relay at all below L3
    from edge.api.gateway import create_app
    from edge.api.handlers import Handlers
    from edge.inference.alert_router import AlertRouter
    from edge.inference.decision import get_default_policy, serving_models_dir
    from edge.inference.explainer import SHAPExplainer
    from edge.inference.flow_ingestor import FlowIngestor
    from edge.inference.scoring_engine import ScoringEngine

    pol = get_default_policy()
    if cfg == "L0":
        pol = replace(pol, rule_detectors_enabled=False)
    se = ScoringEngine(models_dir=serving_models_dir(pol), policy=pol)
    ex = SHAPExplainer(policy=pol) if cfg in ("L2", "L3") else None
    router = AlertRouter(log_dir=tempfile.mkdtemp(prefix=f"x3_{cfg}_"), max_alerts_per_minute=10**9)
    h = Handlers(flow_ingestor=FlowIngestor(), scoring_engine=se, explainer=ex, alert_router=router)
    h.set_model_versions({"x3_config": cfg, "rule_detectors_enabled": pol.rule_detectors_enabled,
                          "shap": ex.describe() if ex else None})
    app = create_app(skip_pipeline=True, handlers_override=h)
    app.router.on_shutdown.append(router.close)
    return app
