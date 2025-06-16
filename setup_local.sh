#!/usr/bin/env bash
set -e

echo "🛠️  BUILDING DOCKER IMAGE..."
docker compose build

echo "🔄  STARTING CONTAINER..."
docker compose up -d

echo "🧪  RUNNING SMOKE TEST (CUDA)..."
docker compose exec omnia-ia python - <<'PY'
import torch, time
t0 = time.time()
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("→ warm-up...")
for _ in range(5):
    torch.randn(4096, 4096, device="cuda").mm(torch.randn(4096, 4096, device="cuda"))
torch.cuda.synchronize()
print("elapsed s:", round(time.time() - t0, 2))
PY
