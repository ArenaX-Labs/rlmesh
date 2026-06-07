# Third-Party Notices

RLMesh is licensed under MIT OR Apache-2.0. This file lists selected
third-party components with bundled assets, bundled data, or non-standard
license terms that are handled explicitly for the RLMesh Python wheel and
native extension.

This is not a complete software bill of materials. Release wheels also include
a generated CycloneDX SBOM under `rlmesh-*.dist-info/sboms/`.

## epaint_default_fonts 0.34.3

- Path: `rlmesh-cli -> eframe default_fonts -> egui/epaint -> epaint_default_fonts`
- Use: default UI font assets for the native render viewer.
- License expression: `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0`
- Included license texts:
  - `third_party_licenses/epaint_default_fonts-0.34.3/OFL.txt`
  - `third_party_licenses/epaint_default_fonts-0.34.3/UFL.txt`
  - `third_party_licenses/epaint_default_fonts-0.34.3/Hack-Regular.txt`
  - `third_party_licenses/epaint_default_fonts-0.34.3/emoji-icon-font-mit-license.txt`

## unicode_names2 1.3.0

- Path: `pyo3-stub-gen -> rustpython-parser -> unicode_names2`
- Use: Unicode name data used by the native Python stub generation stack.
- License expression: `(MIT OR Apache-2.0) AND Unicode-DFS-2016`
- Included license texts:
  - `third_party_licenses/unicode_names2-1.3.0/LICENSE-MIT`
  - `third_party_licenses/unicode_names2-1.3.0/LICENSE-APACHE`
  - `third_party_licenses/unicode_names2-1.3.0/LICENSE-UNICODE`
