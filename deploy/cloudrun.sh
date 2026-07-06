#!/usr/bin/env bash
# Deploy polyrag to Google Cloud Run (fits the always-free tier for demo traffic).
#
# Prereqs:
#   gcloud auth login && gcloud config set project <PROJECT_ID>
#   polyrag ingest && polyrag index      # bake the index into the image
#   A GROQ_API_KEY in your shell env (never committed).
#
# Usage: ./deploy/cloudrun.sh <PROJECT_ID> [REGION]
set -euo pipefail

PROJECT_ID="${1:?usage: ./deploy/cloudrun.sh <PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE="polyrag"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/polyrag/polyrag:latest"

: "${GROQ_API_KEY:?Set GROQ_API_KEY in your environment before deploying}"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT_ID"

gcloud artifacts repositories create polyrag \
  --repository-format=docker --location="$REGION" --project "$PROJECT_ID" 2>/dev/null || true

# Build remotely with Cloud Build (no local Docker needed), then deploy.
gcloud builds submit --tag "$IMAGE" --project "$PROJECT_ID"

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 2 \
  --set-env-vars "GROQ_API_KEY=${GROQ_API_KEY},EMBED_DEVICE=cpu"

echo "Deployed. URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT_ID" --format 'value(status.url)'
