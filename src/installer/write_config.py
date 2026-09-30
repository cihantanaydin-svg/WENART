import json, os, sys
p, price, vlm, skip = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] == "1"
tier, agent_model, skip_llm, gen3d = sys.argv[5], sys.argv[6], sys.argv[7] == "1", sys.argv[8] == "1"
cfg = json.load(open(p)) if os.path.exists(p) else {}
cfg.update(gpu_price_per_hour=price, vlm_model=vlm, pod_tier=tier)
cfg.setdefault("polish", {})["enabled"] = not skip and cfg.get("polish", {}).get("enabled", True)
cfg.setdefault("llm", {})["model"] = agent_model
cfg.setdefault("agent", {})["backend"] = "rules" if skip_llm else "llm"
cfg.setdefault("gen3d", {})["enabled"] = gen3d
json.dump(cfg, open(p, "w"), indent=1)
print("settings file:", p, cfg)
