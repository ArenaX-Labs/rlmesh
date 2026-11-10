# Tether conformance vectors

`vectors.json` holds the byte-exact conformance cases. Every implementation of
the tether runs it: this crate (`cargo nextest run -p rlmesh-tether --features board --test
vectors`) and the Arduino library's host runner (`mise run test:tether:cpp`).

`vectors.rs` generates the file from this crate and checks that the committed
copy matches. Each case also asserts what it is meant to show (a rejection's
error, a round trip), so a regression in the crate can't regenerate silently.
Regenerate only when the contract changes, then format the file:

```bash
mise run dev:heavy -- env RLMESH_REGENERATE_TETHER_VECTORS=1 cargo nextest run -p rlmesh-tether --features board --test vectors -E 'test(the_file_matches_this_build)'
mise run fmt:config
```

## Schema

Bytes are lowercase hex strings. Integers are JSON numbers. Every case has a
`name`. The top level is an object with `"schema": 1` and these sections:

| Section                      | Case fields                                                             | Check                                                                                                                                                  |
| ---------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `crc`                        | `data`, `crc16`, `crc32`                                                | CRC-16/CCITT-FALSE and CRC-32/ISO-HDLC of `data`.                                                                                                      |
| `cobs_encode`                | `decoded`, `encoded`                                                    | `decoded` encodes to `encoded` (no delimiter), in the pinned form, and decodes back.                                                                   |
| `cobs_decode`                | `wire`, `rx_max`, `results`                                             | Feeding `wire` (delimiters included) to a decoder with an `rx_max`-byte buffer yields `results` in order.                                              |
| `frames`                     | `wire`, `rx_max`, `results`                                             | As `cobs_decode`, with each decoded frame then parsed (size and CRC).                                                                                  |
| `messages`                   | `seq`, `msg`, `wire`, optional `dense_len`                              | Writing `msg` with `seq` gives exactly `wire` (with its trailing `00`); decoding `wire` at the end its type is meant for gives `msg` back.             |
| `rejections`                 | `receiver`, `type`, `payload`, `error`, optional `version`, `dense_len` | Decoding a frame of `type` with `payload` at `receiver` (`"bridge"` or `"board"`) fails with `error`; a `version` error also reports the version byte. |
| `write_rejections`           | `msg`, `error`                                                          | Writing `msg` fails with `error` and writes nothing.                                                                                                   |
| `pack`                       | `fields`, `offsets`, `lens`, `aligned`, `dense`                         | `to_dense(aligned) == dense` and `to_aligned(dense) == aligned` (padding zeroed). Bridge only: a board never sees the aligned form.                    |
| `provision`                  | `seq`, `msg`, `wire`                                                    | As `messages`, for USB provisioning (`provision::write`/`decode`): `wire` starts with its leading `00`. A networked board and the daemon only.         |
| `provision_rejections`       | `receiver`, `type`, `payload`, `error`, optional `version`              | As `rejections`; the daemon receives as `"bridge"`.                                                                                                    |
| `provision_write_rejections` | `msg`, `error`                                                          | As `write_rejections`.                                                                                                                                 |

A result in `results` is one of:

- `{"ok": <hex>}`: a decoded frame (`cobs_decode`);
- `{"frame": {"type": n, "seq": n, "payload": <hex>}}`: a valid frame (`frames`);
- `{"error": e}`, with `e` one of `overflow`, `truncated` (COBS) or `size`, `crc` (frame).

Delimiters with nothing before them yield no result.

`dense_len` is an object from a part index (as a string) to that part's dense
value length; a part not listed is unknown. It is the sample length for
`STATE` at the bridge and the command length for `COMMAND` at the board.

`msg` is an object with `type` (the message name: `HELLO`, `MAP`, `STATE`,
`STATUS`, `NOTE`, `GET_MAP`, `ATTACH`, `COMMAND`, `SAFE`, `PING`, `DETACH`)
and the message field names. `board_id`, `name`, `data`, `text` and `value` are
hex.

A provisioning `msg` has `type` `GET_DEVICE`, `DEVICE_INFO`, `PROVISION` or
`PROVISIONED` and the provisioning field names. `device_key`, `host_key`, `host_id`,
`name` and each of `addresses` are hex.

`error` is one of `unknown_type`, `magic`, `version`, `length`, `reserved`,
`range`, `unknown_part`, `direction`, `utf8`.

A runner that doesn't understand `schema` must refuse the file.

Provisioning codec v2 assigns the robot name over USB. Its `PROVISION` payload is
`version:u8=2 | flags:u8 | host_key:32 | host_id_len:u8 | host_id | name_len:u8 |
name | address_count:u8 | (address_len:u8 | address)*`. The name is a participant
id (`[A-Za-z0-9._-]`, 1–128 bytes); host ids remain at most 32 bytes and the
whole payload must fit 252 bytes. `DEVICE_INFO` keeps its length-prefixed name
field, now the stored name, empty before provisioning. v1 records have no
assigned name and are refused: provision again.

`test:tether:cpp` also runs the ESP32 provisioning vectors through the device
C ABI: C++ constructs and frames records independently, then checks the exact
stored blob and the Rust responder's CRC confirmation. Invalid record vectors
must be refused by the ABI.
