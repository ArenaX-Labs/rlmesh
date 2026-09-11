# Third-Party Notices

RLMesh is licensed under MIT OR Apache-2.0. This file lists selected third-party components with bundled assets, bundled data, or non-standard license terms that are handled explicitly for the RLMesh Python wheel and native extension.

This is not a complete software bill of materials. Release wheels also include a generated CycloneDX SBOM under `rlmesh-*.dist-info/sboms/`.

## Independent JPEG Group

The native extension links `jpeg-encoder`, which implements the declared
`rlmesh.adapters.Image.jpeg_quality` round-trip from the Independent JPEG
Group's quantization and Huffman tables and is licensed
`(MIT OR Apache-2.0) AND IJG`. As the IJG terms require:

> This software is based in part on the work of the Independent JPEG Group.

No third-party components currently require explicit bundled license texts.
