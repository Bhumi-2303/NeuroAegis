#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# fetch_models.sh – Ensure model artifacts are present before API start
#
# Usage (local):   bash scripts/fetch_models.sh
# Usage (Docker):  Runs automatically via docker-compose command
#
# The script checks each configured dataset directory for a sentinel
# file (metadata.json).  If missing it will:
#   1. Try a remote download when MODEL_DOWNLOAD_URL is set, or
#   2. Fall back to copying from the repo-local source directories.
#
# Environment variables:
#   MODELS_DIR           – Target models root  (default: apps/api/models)
#   MODEL_DOWNLOAD_URL   – Optional base URL for remote model tarballs
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────
# When run inside Docker the WORKDIR is /app; locally we resolve
# relative to the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -d "/app/models" ]]; then
    # Running inside the Docker container
    MODELS_DIR="${MODELS_DIR:-/app/models}"
    REPO_ROOT="/app"
else
    # Running locally from repo root
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/apps/api/models}"
fi

# ── Dataset definitions ──────────────────────────────────────────────
# Each entry:  dataset_name|sentinel_file|space-separated required files
DATASETS=(
    "bonn|metadata.json|metadata.json feature_names.json selected_features.json final_lightgbm_full_dataset.pkl"
    "chbmit|metadata.json|metadata.json feature_names.json selected_features.json lightgbm_patient_wise.pkl"
)

# ── Colours (disable when not in a terminal) ─────────────────────────
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; RED=''; NC=''
fi

log_info()  { echo -e "${GREEN}[fetch_models]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[fetch_models]${NC} $*"; }
log_error() { echo -e "${RED}[fetch_models]${NC} $*"; }

# ── Remote download helper (placeholder) ─────────────────────────────
fetch_remote() {
    local dataset="$1"
    local target_dir="$2"

    if [[ -z "${MODEL_DOWNLOAD_URL:-}" ]]; then
        return 1  # No URL configured – skip to local copy
    fi

    local url="${MODEL_DOWNLOAD_URL%/}/${dataset}.tar.gz"
    log_info "Downloading ${dataset} models from ${url} ..."

    if command -v curl &>/dev/null; then
        curl -fSL --retry 3 --retry-delay 5 "${url}" | tar -xz -C "${target_dir}"
    elif command -v wget &>/dev/null; then
        wget -qO- "${url}" | tar -xz -C "${target_dir}"
    else
        log_warn "Neither curl nor wget available – cannot download."
        return 1
    fi
}

# ── Local copy helper ────────────────────────────────────────────────
copy_local() {
    local dataset="$1"
    local target_dir="$2"
    local source_dir="${REPO_ROOT}/apps/api/models/${dataset}"

    # Inside Docker the source and target may be the same (mounted volume)
    if [[ "$(realpath "${source_dir}" 2>/dev/null)" == "$(realpath "${target_dir}" 2>/dev/null)" ]]; then
        log_info "Source and target are the same volume – nothing to copy for ${dataset}."
        return 0
    fi

    if [[ -d "${source_dir}" ]]; then
        log_info "Copying ${dataset} artifacts from ${source_dir} → ${target_dir} ..."
        cp -a "${source_dir}/." "${target_dir}/"
        return 0
    fi

    log_warn "Local source ${source_dir} not found."
    return 1
}

# ── Main loop ────────────────────────────────────────────────────────
ERRORS=0

for entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset sentinel required_files <<< "${entry}"

    target_dir="${MODELS_DIR}/${dataset}"
    mkdir -p "${target_dir}"

    sentinel_path="${target_dir}/${sentinel}"

    if [[ -f "${sentinel_path}" ]]; then
        log_info "✓ ${dataset}: sentinel '${sentinel}' found – models present."
        continue
    fi

    log_warn "✗ ${dataset}: sentinel '${sentinel}' missing – attempting to fetch ..."

    # Try remote first, fall back to local copy
    if ! fetch_remote "${dataset}" "${target_dir}"; then
        if ! copy_local "${dataset}" "${target_dir}"; then
            log_error "Could not provision models for ${dataset}."
            ERRORS=$((ERRORS + 1))
            continue
        fi
    fi

    # Verify sentinel exists after provisioning
    if [[ -f "${sentinel_path}" ]]; then
        log_info "✓ ${dataset}: provisioned successfully."
    else
        log_error "✗ ${dataset}: sentinel still missing after provisioning."
        ERRORS=$((ERRORS + 1))
    fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
if [[ ${ERRORS} -gt 0 ]]; then
    log_error "${ERRORS} dataset(s) could not be provisioned. API will start in degraded mode."
    # Exit 0 so the container still starts (graceful degradation)
    exit 0
else
    log_info "All model artifacts verified. Starting API ..."
    exit 0
fi
