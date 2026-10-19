# Sets KEY and HOST for calling the API with curl.
# Run it with "source" (not bash), so the variables stay in your terminal:
#
#   source scripts/api_env.sh
#   curl -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
#        -d '{"sharepoint_folder": "Patient Encounters/Physician Notes/Pending"}' \
#        http://$HOST/dr-nurse-match

KEY=$(uv run python -c "from _SECRETS._secrets import API_TOKEN; print(API_TOKEN)")
HOST=localhost:8010
