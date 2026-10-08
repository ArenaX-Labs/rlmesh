# Sourced by mise (`_.source` in mise.toml); exports nothing unless opted in.
#
# On a machine with RLMESH_BUILDER=depot, point sccache at Depot Cache with the
# credentials `depot login` stored, so local builds and CI share one cache.
# Depot runners preset SCCACHE_WEBDAV_*, so an existing endpoint always wins.
# A user token covers several organizations, hence org ID as the username.

if [ "${RLMESH_BUILDER:-}" = depot ] && [ -z "${SCCACHE_WEBDAV_ENDPOINT:-}" ]; then
  _depot_config="${XDG_CONFIG_HOME:-$HOME/.config}/depot/depot.yaml"
  _depot_token="${DEPOT_TOKEN:-}"
  _depot_org="${DEPOT_ORG_ID:-}"
  if [ -r "$_depot_config" ]; then
    [ -n "$_depot_token" ] || _depot_token=$(sed -n 's/^api_token:[[:space:]]*//p' "$_depot_config")
    [ -n "$_depot_org" ] || _depot_org=$(sed -n 's/^org_id:[[:space:]]*//p' "$_depot_config")
  fi
  if [ -n "$_depot_token" ] && [ -n "$_depot_org" ]; then
    export SCCACHE_WEBDAV_ENDPOINT=https://cache.depot.dev
    export SCCACHE_WEBDAV_USERNAME="$_depot_org"
    export SCCACHE_WEBDAV_PASSWORD="$_depot_token"
  fi
  unset _depot_config _depot_token _depot_org
fi
