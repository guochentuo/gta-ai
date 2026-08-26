#!/usr/bin/env bash
set -euo pipefail

readonly input_file="${1:?用法：render-keepalived-haproxy-track.sh INPUT OUTPUT}"
readonly output_file="${2:?用法：render-keepalived-haproxy-track.sh INPUT OUTPUT}"

if grep -q 'vrrp_script gta_haproxy_es_web' "$input_file"; then
  cp -- "$input_file" "$output_file"
  exit 0
fi

LC_ALL=C awk '
BEGIN {
    in_global_defs = 0
    in_es_web = 0
    script_added = 0
    tracking_added = 0
}

{
    if ($0 ~ /^vrrp_instance[[:space:]]+VI_ES_WEB[[:space:]]*\{/) {
        in_es_web = 1
    }

    if (in_es_web && !tracking_added &&
        $0 ~ /^[[:space:]]+virtual_ipaddress[[:space:]]*\{/) {
        print "    track_script {"
        print "        gta_haproxy_es_web"
        print "    }"
        print ""
        tracking_added = 1
    }

    print

    if ($0 ~ /^global_defs[[:space:]]*\{/) {
        in_global_defs = 1
    } else if (in_global_defs && $0 ~ /^}/) {
        print ""
        print "vrrp_script gta_haproxy_es_web {"
        print "    script \"/usr/bin/systemctl is-active --quiet haproxy\""
        print "    interval 2"
        print "    timeout 2"
        print "    fall 3"
        print "    rise 2"
        print "    weight -60"
        print "}"
        in_global_defs = 0
        script_added = 1
    }

    if (in_es_web && $0 ~ /^}/) {
        in_es_web = 0
    }
}

END {
    if (script_added != 1 || tracking_added != 1) {
        exit 42
    }
}
' "$input_file" >"$output_file"
