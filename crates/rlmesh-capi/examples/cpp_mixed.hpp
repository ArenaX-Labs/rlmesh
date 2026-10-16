// Shared by translation units built at different standards in one binary.
#pragma once
#include <rlmesh.hpp>

rlmesh::RunOptions mixed_options(rlmesh::RunOptions options);
rlmesh::Span<const int> mixed_view(rlmesh::Span<const int> values);
rlmesh::adapters::ImageTag mixed_tag(rlmesh::adapters::ImageTag tag);
rlmesh::Model mixed_model();
size_t mixed_run_options_size();
