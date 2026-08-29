#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONAR_HOST_URL="${SONAR_HOST_URL:-http://localhost:9000}"
SONAR_SCANNER_IMAGE="${SONAR_SCANNER_IMAGE:-sonarsource/sonar-scanner-cli:12.1.0.3233_8.0.1}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: $name no está definido." >&2
    exit 1
  fi
}

require_env SONAR_BACKEND_TOKEN
require_env SONAR_FRONTEND_TOKEN

section() {
  printf '\n========================================\n%s\n========================================\n' "$1"
}

section "Backend: lint, tests y cobertura"
(
  cd "$ROOT_DIR/backend"
  docker compose exec -T backend ruff check app tests
  docker compose exec -T backend \
    pytest \
    --cov=app \
    --cov-report=term-missing \
    --cov-report=xml:coverage/coverage.xml
)

section "Frontend: lint, tests, build y cobertura"
(
  cd "$ROOT_DIR/frontend"
  npm run lint
  npm test
  npm run build
  npm run test:coverage
)

run_sonar() {
  local project="$1"
  local token="$2"

  section "SonarQube: $project"

  docker run --rm \
    --network host \
    -e SONAR_HOST_URL="$SONAR_HOST_URL" \
    -e SONAR_TOKEN="$token" \
    -v "$ROOT_DIR:/usr/src" \
    -w "/usr/src/$project" \
    "$SONAR_SCANNER_IMAGE" \
    -Dsonar.qualitygate.wait=true \
    -Dsonar.qualitygate.timeout=300
}

run_sonar backend "$SONAR_BACKEND_TOKEN"
run_sonar frontend "$SONAR_FRONTEND_TOKEN"

section "QUALITY ANALYSIS PASSED"
