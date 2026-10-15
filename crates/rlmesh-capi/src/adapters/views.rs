//! Immutable lent C trees. All allocations remain at fixed addresses.
#![allow(unsafe_code)]
use super::*;
pub(crate) fn view_rotation(arena: &mut Arena, f: &a::Rotation) -> RLMeshRotation {
    let mut v: RLMeshRotation = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.encoding = number_rotation_encoding(f.encoding);
    v.value_len = f.value.len();
    v.value = arena.slice(f.value.clone());
    v
}
pub(crate) fn view_custom_encoding(
    arena: &mut Arena,
    f: &a::CustomEncoding,
) -> RLMeshCustomEncoding {
    let mut v: RLMeshCustomEncoding = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.base = number_rotation_encoding(f.base);
    v.from_base = arena.arm(f.from_base.as_ref());
    v.to_base = arena.arm(f.to_base.as_ref());
    v.name = arena.text(&f.name);
    v
}
pub(crate) fn view_image(arena: &mut Arena, f: &a::Image) -> RLMeshImage {
    let mut v: RLMeshImage = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    v.has_height = f.height.is_some();
    v.height = f.height.unwrap_or_default();
    v.has_width = f.width.is_some();
    v.width = f.width.unwrap_or_default();
    v.layout = number_image_layout(f.layout);
    v.has_channels = f.channels.is_some();
    v.channels = f.channels.unwrap_or_default();
    let (number, other) = arena.tolerant(&f.dtype, number_d_type);
    v.dtype = number;
    v.dtype_other = other;
    match f.normalize {
        a::Normalize::Off => v.normalize = 1,
        a::Normalize::Unit => v.normalize = 2,
        a::Normalize::Range(lo, hi) => {
            v.normalize = 3;
            v.normalize_range = [lo, hi];
        }
    }
    v.lead_dims = f.lead_dims;
    v.upside_down = f.upside_down;
    let (number, other) = arena.tolerant(&f.resample, number_resample);
    v.resample = number;
    v.resample_other = other;
    v.allow_upscale = f.allow_upscale;
    if let Some(items) = &f.fit {
        let (values, others) = arena.accept(items, number_fit_mode);
        v.fit_len = items.len();
        v.fit = values;
        v.fit_other = others;
    }
    v.optional = f.optional;
    v.has_fill = f.fill.is_some();
    v.fill = f.fill.unwrap_or_default();
    v.stack = f.stack;
    v.has_crop = f.crop.is_some();
    v.crop = f.crop.unwrap_or_default();
    v.has_crop_area = f.crop_area.is_some();
    v.crop_area = f.crop_area.unwrap_or_default();
    let (number, other) = arena.tolerant(&f.crop_mode, number_crop_mode);
    v.crop_mode = number;
    v.crop_mode_other = other;
    v.has_jpeg_quality = f.jpeg_quality.is_some();
    v.jpeg_quality = f.jpeg_quality.unwrap_or_default();
    let (number, other) = arena.tolerant(&f.channel_order, number_channel_order);
    v.channel_order = number;
    v.channel_order_other = other;
    v.has_render = f.render.is_some();
    let pair = f.render.unwrap_or_default();
    v.render = [pair.0, pair.1];
    if let Some(items) = &f.offsets {
        v.offsets_len = items.len();
        v.offsets = arena.slice(items.clone());
    }
    v.stack_pad = number_stack_pad(f.stack_pad);
    v.part = arena.optional_text(f.part.as_deref());
    v
}
pub(crate) fn view_state(arena: &mut Arena, f: &a::State) -> RLMeshState {
    let mut v: RLMeshState = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    if let Some(e) = &f.encoding {
        match e {
            a::StateEncoding::Custom(e) => {
                let view = view_custom_encoding(arena, e);
                v.custom_encoding = arena.keep(view);
            }
            a::StateEncoding::Native(items) => {
                let (values, others) = arena.accept(items, number_rotation_encoding);
                v.encoding_len = items.len();
                v.encoding = values;
                v.encoding_other = others;
            }
        }
    }
    v.has_dim = f.dim.is_some();
    v.dim = f.dim.unwrap_or_default();
    v.has_index = f.index.is_some();
    v.index = f.index.unwrap_or_default();
    v.optional = f.optional;
    v.has_range = f.range.is_some();
    let pair = f.range.unwrap_or_default();
    v.range = [pair.0, pair.1];
    v.has_pad_to = f.pad_to.is_some();
    v.pad_to = f.pad_to.unwrap_or_default();
    let (number, other) = arena.tolerant(&f.dtype, number_d_type);
    v.dtype = number;
    v.dtype_other = other;
    if let Some(items) = &f.reshape {
        v.reshape_len = items.len();
        v.reshape = arena.slice(items.clone());
    }
    v.container = number_state_container(f.container);
    v.fill = f.fill;
    if let Some(leaf) = &f.post_rotate {
        let view = view_rotation(arena, leaf);
        v.post_rotate = arena.keep(view);
    }
    if let Some(axis) = &f.scale {
        match axis {
            a::AxisValue::Scalar(value) => {
                v.has_scale = true;
                v.scale = *value;
            }
            a::AxisValue::PerAxis(values) => {
                v.axis_scale_len = values.len();
                v.axis_scale = arena.slice(values.clone());
            }
        }
    }
    if let Some(axis) = &f.offset {
        match axis {
            a::AxisValue::Scalar(value) => {
                v.has_offset = true;
                v.offset = *value;
            }
            a::AxisValue::PerAxis(values) => {
                v.axis_offset_len = values.len();
                v.axis_offset = arena.slice(values.clone());
            }
        }
    }
    v.has_frame = f.frame.is_some();
    if v.has_frame {
        let (number, other) =
            arena.tolerant(f.frame.as_ref().expect("present value"), number_frame);
        v.frame = number;
        v.frame_other = other;
    }
    if let Some(items) = &f.provenance {
        let (values, others) = arena.accept(items, number_provenance);
        v.provenance_len = items.len();
        v.provenance = values;
        v.provenance_other = others;
    }
    v.part = arena.optional_text(f.part.as_deref());
    if let Some(items) = &f.labels {
        v.labels_len = items.len();
        v.labels = arena.strings(items);
    }
    v.source = number_state_source(f.source);
    v.has_clip = f.clip.is_some();
    let pair = f.clip.unwrap_or_default();
    v.clip = [pair.0, pair.1];
    v
}
pub(crate) fn view_concat(arena: &mut Arena, f: &a::Concat) -> RLMeshConcat {
    let mut v: RLMeshConcat = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    let items = f
        .parts
        .iter()
        .map(|part| view_part(arena, part))
        .collect::<Vec<_>>();
    v.parts_len = items.len();
    v.parts = arena.slice(items);
    v.has_pad_to = f.pad_to.is_some();
    v.pad_to = f.pad_to.unwrap_or_default();
    let (number, other) = arena.tolerant(&f.dtype, number_d_type);
    v.dtype = number;
    v.dtype_other = other;
    if let Some(items) = &f.reshape {
        v.reshape_len = items.len();
        v.reshape = arena.slice(items.clone());
    }
    v.container = number_state_container(f.container);
    v.has_clip = f.clip.is_some();
    let pair = f.clip.unwrap_or_default();
    v.clip = [pair.0, pair.1];
    v
}
pub(crate) fn view_constant(_arena: &mut Arena, f: &a::Constant) -> RLMeshConstant {
    let mut v: RLMeshConstant = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.dim = f.dim;
    v.fill = f.fill;
    v
}
pub(crate) fn view_text(arena: &mut Arena, f: &a::Text) -> RLMeshAdaptersText {
    let mut v: RLMeshAdaptersText = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    v.container = number_text_container(f.container);
    v.has_fill = f.fill.is_some();
    if let Some(fill) = &f.fill {
        v.fill = arena.str_view(fill);
    }
    v
}
pub(crate) fn view_custom(arena: &mut Arena, f: &a::Custom) -> RLMeshCustom {
    let mut v: RLMeshCustom = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    if let Some(t) = &f.transform {
        v.transform = arena.keep(RLMeshObsTransform(t.clone()));
    }
    v.entrypoint = arena.optional_text(f.entrypoint.as_deref());
    v
}
pub(crate) fn view_actuator(arena: &mut Arena, f: &a::Actuator) -> RLMeshActuator {
    let mut v: RLMeshActuator = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.optional_text(f.role.as_deref());
    v.dim = f.dim;
    if let Some(e) = &f.encoding {
        match e {
            a::ActionEncoding::Custom(e) => {
                let view = view_custom_encoding(arena, e);
                v.custom_encoding = arena.keep(view);
            }
            a::ActionEncoding::Native(e) => {
                v.has_encoding = true;
                v.encoding = number_rotation_encoding(*e);
            }
        }
    }
    v.has_range = f.range.is_some();
    let pair = f.range.unwrap_or_default();
    v.range = [pair.0, pair.1];
    v.binary = f.binary;
    if let Some(axis) = &f.scale {
        match axis {
            a::AxisValue::Scalar(value) => {
                v.has_scale = true;
                v.scale = *value;
            }
            a::AxisValue::PerAxis(values) => {
                v.axis_scale_len = values.len();
                v.axis_scale = arena.slice(values.clone());
            }
        }
    }
    v.invert = f.invert;
    v.has_threshold = f.threshold.is_some();
    v.threshold = f.threshold.unwrap_or_default();
    v.clip = f.clip;
    v.fill = f.fill;
    v.optional = f.optional;
    v.has_frame = f.frame.is_some();
    if v.has_frame {
        let (number, other) =
            arena.tolerant(f.frame.as_ref().expect("present value"), number_frame);
        v.frame = number;
        v.frame_other = other;
    }
    v.has_reference = f.reference.is_some();
    if v.has_reference {
        let (number, other) = arena.tolerant(
            f.reference.as_ref().expect("present value"),
            number_reference,
        );
        v.reference = number;
        v.reference_other = other;
    }
    v.part = arena.optional_text(f.part.as_deref());
    if let Some(axis) = &f.offset {
        match axis {
            a::AxisValue::Scalar(value) => {
                v.has_offset = true;
                v.offset = *value;
            }
            a::AxisValue::PerAxis(values) => {
                v.axis_offset_len = values.len();
                v.axis_offset = arena.slice(values.clone());
            }
        }
    }
    if let Some(items) = &f.labels {
        v.labels_len = items.len();
        v.labels = arena.strings(items);
    }
    v
}
pub(crate) fn view_action(arena: &mut Arena, f: &a::Action) -> RLMeshAction {
    let mut v: RLMeshAction = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    let items = f
        .components
        .iter()
        .map(|leaf| {
            let view = view_actuator(arena, leaf);
            arena.keep(view)
        })
        .collect::<Vec<_>>();
    v.components_len = items.len();
    v.components = arena.slice(items);
    v.has_clip = f.clip.is_some();
    let pair = f.clip.unwrap_or_default();
    v.clip = [pair.0, pair.1];
    v
}
pub(crate) fn view_image_tag(arena: &mut Arena, f: &a::ImageTag) -> RLMeshImageTag {
    let mut v: RLMeshImageTag = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    v.layout = number_image_layout(f.layout);
    v.upside_down = f.upside_down;
    v.part = arena.optional_text(f.part.as_deref());
    v
}
pub(crate) fn view_state_tag(arena: &mut Arena, f: &a::StateTag) -> RLMeshStateTag {
    let mut v: RLMeshStateTag = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    if let Some(items) = &f.encoding {
        let (values, others) = arena.accept(items, number_rotation_encoding);
        v.encoding_len = items.len();
        v.encoding = values;
        v.encoding_other = others;
    }
    v.has_range = f.range.is_some();
    let pair = f.range.unwrap_or_default();
    v.range = [pair.0, pair.1];
    v.has_frame = f.frame.is_some();
    if v.has_frame {
        let (number, other) =
            arena.tolerant(f.frame.as_ref().expect("present value"), number_frame);
        v.frame = number;
        v.frame_other = other;
    }
    v.has_provenance = f.provenance.is_some();
    if v.has_provenance {
        let (number, other) = arena.tolerant(
            f.provenance.as_ref().expect("present value"),
            number_provenance,
        );
        v.provenance = number;
        v.provenance_other = other;
    }
    v.part = arena.optional_text(f.part.as_deref());
    if let Some(items) = &f.labels {
        v.labels_len = items.len();
        v.labels = arena.strings(items);
    }
    v
}
pub(crate) fn view_text_tag(arena: &mut Arena, f: &a::TextTag) -> RLMeshTextTag {
    let mut v: RLMeshTextTag = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.text(&f.role);
    v
}
pub(crate) fn view_field(arena: &mut Arena, f: &a::Field) -> RLMeshField {
    let mut v: RLMeshField = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.role = arena.optional_text(f.role.as_deref());
    v.dim = f.dim;
    if let Some(items) = &f.encoding {
        let (values, others) = arena.accept(items, number_rotation_encoding);
        v.encoding_len = items.len();
        v.encoding = values;
        v.encoding_other = others;
    }
    v.has_range = f.range.is_some();
    let pair = f.range.unwrap_or_default();
    v.range = [pair.0, pair.1];
    v.has_frame = f.frame.is_some();
    if v.has_frame {
        let (number, other) =
            arena.tolerant(f.frame.as_ref().expect("present value"), number_frame);
        v.frame = number;
        v.frame_other = other;
    }
    v.has_provenance = f.provenance.is_some();
    if v.has_provenance {
        let (number, other) = arena.tolerant(
            f.provenance.as_ref().expect("present value"),
            number_provenance,
        );
        v.provenance = number;
        v.provenance_other = other;
    }
    v.part = arena.optional_text(f.part.as_deref());
    if let Some(items) = &f.labels {
        v.labels_len = items.len();
        v.labels = arena.strings(items);
    }
    v
}
pub(crate) fn view_split(arena: &mut Arena, f: &a::Split) -> RLMeshSplit {
    let mut v: RLMeshSplit = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    let items = f
        .fields
        .iter()
        .map(|leaf| {
            let view = view_field(arena, leaf);
            arena.keep(view)
        })
        .collect::<Vec<_>>();
    v.fields_len = items.len();
    v.fields = arena.slice(items);
    v
}

pub(crate) fn view_observation_roles(
    arena: &mut Arena,
    f: &a::ObservationRoles,
) -> RLMeshObservationRoles {
    let mut v: RLMeshObservationRoles = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.images_len = f.images.len();
    v.images = arena.strings(&f.images);
    v.states_len = f.states.len();
    v.states = arena.strings(&f.states);
    v.texts_len = f.texts.len();
    v.texts = arena.strings(&f.texts);
    v
}
pub(crate) fn view_advisory(arena: &mut Arena, f: &a::Advisory) -> RLMeshAdvisory {
    let mut v: RLMeshAdvisory = unsafe { std::mem::zeroed() };
    v.struct_size = std::mem::size_of_val(&v);
    v.severity = number_advisory_severity(f.severity);
    v.message = arena.text(&f.message);
    v
}

#[derive(Default)]
pub(crate) struct Arena(Vec<Box<dyn std::any::Any>>);
impl Arena {
    pub(crate) fn keep<T: 'static>(&mut self, value: T) -> *const T {
        let boxed = Box::new(value);
        let ptr = std::ptr::from_ref(&*boxed);
        self.0.push(boxed);
        ptr
    }
    fn slice<T: 'static>(&mut self, values: Vec<T>) -> *const T {
        let ptr = values.as_ptr();
        self.keep(values);
        ptr
    }
    fn text(&mut self, text: &str) -> *const c_char {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        self.slice(bytes).cast()
    }
    fn optional_text(&mut self, text: Option<&str>) -> *const c_char {
        text.map_or(std::ptr::null(), |text| self.text(text))
    }
    fn str_view(&mut self, text: &str) -> RLMeshStr {
        let len = text.len();
        RLMeshStr {
            data: self.slice(text.as_bytes().to_vec()).cast(),
            len,
        }
    }
    fn strings(&mut self, items: &[String]) -> *const *const c_char {
        let items = items.iter().map(|s| self.text(s)).collect();
        self.slice(items)
    }
    fn tolerant<E: a::Vocabulary>(
        &mut self,
        value: &a::Tolerant<E>,
        number: fn(E) -> i32,
    ) -> (i32, *const c_char) {
        match value {
            a::Tolerant::Known(e) => (number(*e), std::ptr::null()),
            a::Tolerant::Other(s) => (1000, self.text(s)),
        }
    }
    fn accept<E: a::Vocabulary>(
        &mut self,
        items: &[a::Tolerant<E>],
        number: fn(E) -> i32,
    ) -> (*const i32, *const *const c_char) {
        let (values, others): (Vec<_>, Vec<_>) =
            items.iter().map(|e| self.tolerant(e, number)).unzip();
        (self.slice(values), self.slice(others))
    }
    fn arm(&mut self, arm: Option<&a::EncodingArm>) -> RLMeshEncodingArm {
        let mut out = RLMeshEncodingArm {
            transform: std::ptr::null(),
            entrypoint: std::ptr::null(),
        };
        match arm {
            Some(a::EncodingArm::Transform(t)) => {
                out.transform = self.keep(RLMeshRotationTransform(t.clone()))
            }
            Some(a::EncodingArm::Entrypoint(s)) => out.entrypoint = self.text(s),
            None => {}
        }
        out
    }
}
// SAFETY: these arenas are immutable after construction. Every raw pointer
// refers to an owned allocation in the same arena, and callable leaves contain
// only the facade's Send+Sync shared transforms. No pointer escapes mutably.
unsafe impl Send for SpecView {}
unsafe impl Sync for SpecView {}
unsafe impl Send for TagsView {}
unsafe impl Sync for TagsView {}
pub(crate) struct SpecView {
    pub input: *const RLMeshInputNode,
    pub action: *const RLMeshAction,
    _arena: Arena,
}
impl SpecView {
    pub(crate) fn new(value: &a::ModelSpec) -> Self {
        let mut arena = Arena::default();
        let input = view_input(&mut arena, value.input());
        let input = arena.keep(input);
        let action = view_action(&mut arena, value.output());
        let action = arena.keep(action);
        Self {
            input,
            action,
            _arena: arena,
        }
    }
}
pub(crate) struct TagsView {
    pub observation: *const RLMeshObsNode,
    pub action: *const RLMeshAction,
    pub roles: RLMeshObservationRoles,
    _arena: Arena,
}
impl TagsView {
    pub(crate) fn new(value: &a::EnvTags) -> Self {
        let mut arena = Arena::default();
        let observation = view_obs(&mut arena, value.observation());
        let observation = arena.keep(observation);
        let action = view_action(&mut arena, value.action());
        let action = arena.keep(action);
        let roles = view_observation_roles(&mut arena, &value.observation_roles());
        Self {
            observation,
            action,
            roles,
            _arena: arena,
        }
    }
}
fn view_part(arena: &mut Arena, p: &a::ConcatPart) -> RLMeshConcatPart {
    let mut out = RLMeshConcatPart {
        kind: 0,
        role: std::ptr::null(),
        leaf: std::ptr::null(),
    };
    match p {
        a::ConcatPart::Role(s) => {
            out.kind = 1;
            out.role = arena.text(s)
        }
        a::ConcatPart::State(v) => {
            out.kind = 2;
            let view = view_state(arena, v);
            out.leaf = arena.keep(view).cast()
        }
        a::ConcatPart::Constant(v) => {
            out.kind = 3;
            let view = view_constant(arena, v);
            out.leaf = arena.keep(view).cast()
        }
    }
    out
}
fn view_input(arena: &mut Arena, n: &a::InputNode) -> RLMeshInputNode {
    let mut out = RLMeshInputNode {
        kind: 0,
        leaf: std::ptr::null(),
        keys: std::ptr::null(),
        children: std::ptr::null(),
        children_len: 0,
    };
    match n {
        a::InputNode::Leaf(leaf) => match leaf {
            a::ModelLeaf::Image(v) => {
                out.kind = 1;
                let view = view_image(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ModelLeaf::State(v) => {
                out.kind = 2;
                let view = view_state(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ModelLeaf::Concat(v) => {
                out.kind = 3;
                let view = view_concat(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ModelLeaf::Text(v) => {
                out.kind = 4;
                let view = view_text(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ModelLeaf::Custom(v) => {
                out.kind = 5;
                let view = view_custom(arena, v);
                out.leaf = arena.keep(view).cast()
            }
        },
        a::InputNode::Dict(pairs) => {
            out.kind = 6;
            let keys = pairs.iter().map(|(k, _)| arena.str_view(k)).collect();
            out.keys = arena.slice(keys);
            let children = pairs
                .iter()
                .map(|(_, n)| view_input(arena, n))
                .collect::<Vec<_>>();
            out.children_len = children.len();
            out.children = arena.slice(children)
        }
        a::InputNode::Tuple(items) => {
            out.kind = 7;
            let children = items
                .iter()
                .map(|n| view_input(arena, n))
                .collect::<Vec<_>>();
            out.children_len = children.len();
            out.children = arena.slice(children)
        }
    }
    out
}
fn view_obs(arena: &mut Arena, n: &a::ObsNode) -> RLMeshObsNode {
    let mut out = RLMeshObsNode {
        kind: 0,
        leaf: std::ptr::null(),
        keys: std::ptr::null(),
        children: std::ptr::null(),
        children_len: 0,
    };
    match n {
        a::ObsNode::Leaf(leaf) => match leaf {
            a::ObsLeaf::ImageTag(v) => {
                out.kind = 1;
                let view = view_image_tag(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ObsLeaf::StateTag(v) => {
                out.kind = 2;
                let view = view_state_tag(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ObsLeaf::TextTag(v) => {
                out.kind = 3;
                let view = view_text_tag(arena, v);
                out.leaf = arena.keep(view).cast()
            }
            a::ObsLeaf::Split(v) => {
                out.kind = 4;
                let view = view_split(arena, v);
                out.leaf = arena.keep(view).cast()
            }
        },
        a::ObsNode::Dict(pairs) => {
            out.kind = 5;
            let keys = pairs.iter().map(|(k, _)| arena.str_view(k)).collect();
            out.keys = arena.slice(keys);
            let children = pairs
                .iter()
                .map(|(_, n)| view_obs(arena, n))
                .collect::<Vec<_>>();
            out.children_len = children.len();
            out.children = arena.slice(children)
        }
        a::ObsNode::Tuple(items) => {
            out.kind = 6;
            let children = items.iter().map(|n| view_obs(arena, n)).collect::<Vec<_>>();
            out.children_len = children.len();
            out.children = arena.slice(children)
        }
    }
    out
}
