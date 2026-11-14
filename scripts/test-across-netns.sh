#!/bin/sh
# Runs COMMAND twice, as a session host and as a client on another network:
# RLMESH_TEST_ROLE=host, then RLMESH_TEST_ROLE=client, both with
# RLMESH_TEST_HOST_IP, the host's non-loopback address. Exits with the
# client's status once the host is stopped.
#
# Two network namespaces joined by a veth pair where unprivileged user
# namespaces allow it (no root needed), so the client's packets cross an
# interface the host does not share; else both on this machine, the host on
# its non-loopback address (also with RLMESH_TEST_NO_NETNS=1). Exits 77
# (skip) when it has neither.
#
# RLMESH_TEST_NETEM (tc netem arguments, "delay 150ms") slows the link both
# ways; it needs the namespaces, and exits 77 without them.
#
# usage: scripts/test-across-netns.sh COMMAND [ARG...]
set -eu

HOST_IP=10.199.0.1
CLIENT_IP=10.199.0.2

say() { echo "test-across-netns: $*" >&2; }

run_pair() {
    RLMESH_TEST_HOST_IP=$1
    export RLMESH_TEST_HOST_IP
    shift
    RLMESH_TEST_ROLE=host "$@" &
    host=$!
    trap 'kill "$host" 2>/dev/null || true' EXIT
    status=0
    # shellcheck disable=SC2086 # $CLIENT_PREFIX is a word list or empty
    RLMESH_TEST_ROLE=client $CLIENT_PREFIX "$@" || status=$?
    kill "$host" 2>/dev/null || true
    wait "$host" 2>/dev/null || true
    trap - EXIT
    exit "$status"
}

if [ "${RLMESH_NETNS_INSIDE:-}" = 1 ]; then
    # Root of a fresh user namespace, in the host's network namespace.
    ip link set lo up
    ip link add rlmesh-h type veth peer name rlmesh-c
    ip addr add "$HOST_IP/24" dev rlmesh-h
    ip link set rlmesh-h up
    unshare -n sleep 100000 &
    park=$!
    trap 'kill "$park" 2>/dev/null || true' EXIT
    while [ "$(readlink "/proc/$park/ns/net")" = "$(readlink /proc/self/ns/net)" ]; do
        sleep 0.01
    done
    ip link set rlmesh-c netns "$park"
    nsenter -t "$park" -n sh -c "ip link set lo up && ip addr add $CLIENT_IP/24 dev rlmesh-c && ip link set rlmesh-c up"
    if [ -n "${RLMESH_TEST_NETEM:-}" ]; then
        # shellcheck disable=SC2086 # netem's arguments, split on purpose
        tc qdisc add dev rlmesh-h root netem $RLMESH_TEST_NETEM
        # shellcheck disable=SC2086
        nsenter -t "$park" -n tc qdisc add dev rlmesh-c root netem $RLMESH_TEST_NETEM
    fi
    CLIENT_PREFIX="nsenter -t $park -n --"
    unset RLMESH_NETNS_INSIDE
    (run_pair "$HOST_IP" "$@") && status=0 || status=$?
    kill "$park" 2>/dev/null || true
    trap - EXIT
    exit "$status"
fi

[ "$#" -gt 0 ] || { echo "usage: $0 COMMAND [ARG...]" >&2; exit 64; }

if [ "${RLMESH_TEST_NO_NETNS:-}" != 1 ] && command -v unshare >/dev/null && command -v nsenter >/dev/null && command -v ip >/dev/null &&
    unshare -Urn sh -c 'ip link add rlmesh-p type veth peer name rlmesh-q' >/dev/null 2>&1 &&
    { [ -z "${RLMESH_TEST_NETEM:-}" ] || unshare -Urn sh -c 'ip link add rlmesh-p type veth peer name rlmesh-q && tc qdisc add dev rlmesh-p root netem delay 1ms' >/dev/null 2>&1; }; then
    say "two network namespaces joined by a veth pair ($HOST_IP, $CLIENT_IP)"
    RLMESH_NETNS_INSIDE=1 exec unshare -Urn --fork --kill-child -- sh "$0" "$@"
fi

if [ -n "${RLMESH_TEST_NETEM:-}" ]; then
    say "skipped: a slowed link needs user network namespaces"
    exit 77
fi

# This machine's address on its default route (no packet is sent).
ip=$(ip -4 route get 192.0.2.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p' | head -n 1) || ip=
case "$ip" in
"" | 127.*)
    say "skipped: no user network namespaces and no non-loopback address on this machine"
    exit 77
    ;;
esac
say "no user network namespaces: host and client on this machine's address $ip"
CLIENT_PREFIX=
run_pair "$ip" "$@"
