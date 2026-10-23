using System;
using System.CodeDom.Compiler;
using System.Diagnostics;
using System.Runtime.InteropServices;

[assembly: GeneratedCode("ClangSharp", "21.1.8.4")]

namespace RLMesh.Interop
{
    internal partial struct DLPackVersion
    {
        [NativeTypeName("uint32_t")]
        public uint major;

        [NativeTypeName("uint32_t")]
        public uint minor;
    }

    [NativeTypeName("unsigned int")]
    internal enum DLDeviceType : uint
    {
        kDLCPU = 1,
        kDLCUDA = 2,
        kDLCUDAHost = 3,
        kDLOpenCL = 4,
        kDLVulkan = 7,
        kDLMetal = 8,
        kDLVPI = 9,
        kDLROCM = 10,
        kDLROCMHost = 11,
        kDLExtDev = 12,
        kDLCUDAManaged = 13,
        kDLOneAPI = 14,
        kDLWebGPU = 15,
        kDLHexagon = 16,
        kDLMAIA = 17,
    }

    internal partial struct DLDevice
    {
        public DLDeviceType device_type;

        [NativeTypeName("int32_t")]
        public int device_id;
    }

    [NativeTypeName("unsigned int")]
    internal enum DLDataTypeCode : uint
    {
        kDLInt = 0U,
        kDLUInt = 1U,
        kDLFloat = 2U,
        kDLOpaqueHandle = 3U,
        kDLBfloat = 4U,
        kDLComplex = 5U,
        kDLBool = 6U,
    }

    internal partial struct DLDataType
    {
        [NativeTypeName("uint8_t")]
        public byte code;

        [NativeTypeName("uint8_t")]
        public byte bits;

        [NativeTypeName("uint16_t")]
        public ushort lanes;
    }

    internal unsafe partial struct DLTensor
    {
        public void* data;

        public DLDevice device;

        [NativeTypeName("int32_t")]
        public int ndim;

        public DLDataType dtype;

        [NativeTypeName("int64_t *")]
        public long* shape;

        [NativeTypeName("int64_t *")]
        public long* strides;

        [NativeTypeName("uint64_t")]
        public ulong byte_offset;
    }

    internal unsafe partial struct DLManagedTensor
    {
        public DLTensor dl_tensor;

        public void* manager_ctx;

        [NativeTypeName("void (*)(struct DLManagedTensor *)")]
        public IntPtr deleter;
    }

    internal unsafe partial struct DLManagedTensorVersioned
    {
        public DLPackVersion version;

        public void* manager_ctx;

        [NativeTypeName("void (*)(struct DLManagedTensorVersioned *)")]
        public IntPtr deleter;

        [NativeTypeName("uint64_t")]
        public ulong flags;

        public DLTensor dl_tensor;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshStatus : uint
    {
        RLMESH_OK = 0,
        RLMESH_ERR_INVALID_ARGUMENT = 1,
        RLMESH_ERR_INVALID_VALUE = 2,
        RLMESH_ERR_ENVIRONMENT = 3,
        RLMESH_ERR_MODEL = 4,
        RLMESH_ERR_CONNECTION = 5,
        RLMESH_ERR_TRANSPORT = RLMESH_ERR_CONNECTION,
        RLMESH_ERR_TIMEOUT = 6,
        RLMESH_ERR_PANIC = 7,
        RLMESH_ERR_CANCELLED = 8,
        RLMESH_ERR_PROTOCOL = 9,
        RLMESH_ERR_TYPE = 10,
        RLMESH_ERR_RUNTIME = 11,
        RLMESH_ERR_OS = 12,
        RLMESH_ERR_INTERNAL = 99,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshErrorClass : uint
    {
        RLMESH_ERROR_CLASS_INVALID = 0,
        RLMESH_ERROR_CLASS_RLMESH_EXCEPTION = 1,
        RLMESH_ERROR_CLASS_PROTOCOL_EXCEPTION = 2,
        RLMESH_ERROR_CLASS_ENVIRONMENT_EXCEPTION = 3,
        RLMESH_ERROR_CLASS_RECOVERABLE_ENVIRONMENT_EXCEPTION = 4,
        RLMESH_ERROR_CLASS_MODEL_EXCEPTION = 5,
        RLMESH_ERROR_CLASS_VALUE_ERROR = 6,
        RLMESH_ERROR_CLASS_ADAPTER_RESOLUTION_ERROR = 7,
        RLMESH_ERROR_CLASS_TYPE_ERROR = 8,
        RLMESH_ERROR_CLASS_RUNTIME_ERROR = 9,
        RLMESH_ERROR_CLASS_OS_ERROR = 10,
        RLMESH_ERROR_CLASS_CONNECTION_ERROR = 11,
        RLMESH_ERROR_CLASS_TIMEOUT_ERROR = 12,
        RLMESH_ERROR_CLASS_KEYBOARD_INTERRUPT = 13,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshErrorCode : uint
    {
        RLMESH_ERROR_CODE_UNSPECIFIED = 0,
        RLMESH_ERROR_CODE_TIMEOUT = 1,
        RLMESH_ERROR_CODE_INVALID_ACTION = 2,
        RLMESH_ERROR_CODE_NOT_READY = 3,
        RLMESH_ERROR_CODE_BUSY = 4,
        RLMESH_ERROR_CODE_UNSUPPORTED = 5,
        RLMESH_ERROR_CODE_INTERNAL = 10,
        RLMESH_ERROR_CODE_CRASHED = 11,
        RLMESH_ERROR_CODE_CANCELLED = 12,
        RLMESH_ERROR_CODE_CLOSED = 13,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshModelErrorCode : uint
    {
        RLMESH_MODEL_ERROR_CODE_UNSPECIFIED = 0,
        RLMESH_MODEL_ERROR_CODE_INVALID_REQUEST = 2,
        RLMESH_MODEL_ERROR_CODE_NOT_CONFIGURED = 3,
        RLMESH_MODEL_ERROR_CODE_BUSY = 4,
        RLMESH_MODEL_ERROR_CODE_UNSUPPORTED = 5,
        RLMESH_MODEL_ERROR_CODE_INTERNAL = 10,
        RLMESH_MODEL_ERROR_CODE_CANCELLED = 12,
        RLMESH_MODEL_ERROR_CODE_CLOSED = 13,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshBuildSource : uint
    {
        RLMESH_BUILD_SOURCE_INVALID = 0,
        RLMESH_BUILD_SOURCE_RELEASE = 1,
        RLMESH_BUILD_SOURCE_PACKAGE = 2,
        RLMESH_BUILD_SOURCE_GIT = 3,
    }

    internal unsafe partial struct RLMeshBuildInfo
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* version;

        [NativeTypeName("const char *")]
        public sbyte* protocol_generation;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition_base;

        [NativeTypeName("const char *const *")]
        public sbyte** supported_workflow_editions;

        [NativeTypeName("size_t")]
        public UIntPtr supported_workflow_editions_len;

        [NativeTypeName("const char *")]
        public sbyte* build_cohort;

        public RLMeshBuildSource build_source;

        [NativeTypeName("uint32_t")]
        public uint _reserved;

        [NativeTypeName("const char *")]
        public sbyte* git;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshWarningCategory : uint
    {
        RLMESH_WARNING_CATEGORY_INVALID = 0,
        RLMESH_WARNING_CATEGORY_USER_WARNING = 1,
        RLMESH_WARNING_CATEGORY_RUNTIME_WARNING = 2,
        RLMESH_WARNING_CATEGORY_WORKFLOW_EDITION_WARNING = 3,
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate RLMeshStatus RLMeshWarningHandlerFn(void* user_data, RLMeshWarningCategory category, [NativeTypeName("const char *")] sbyte* message);

    [NativeTypeName("unsigned int")]
    internal enum RLMeshDType : uint
    {
        RLMESH_DTYPE_INVALID = 0,
        RLMESH_DTYPE_BOOL = 1,
        RLMESH_DTYPE_UINT8 = 2,
        RLMESH_DTYPE_UINT16 = 3,
        RLMESH_DTYPE_UINT32 = 4,
        RLMESH_DTYPE_UINT64 = 5,
        RLMESH_DTYPE_INT8 = 6,
        RLMESH_DTYPE_INT16 = 7,
        RLMESH_DTYPE_INT32 = 8,
        RLMESH_DTYPE_INT64 = 9,
        RLMESH_DTYPE_FLOAT16 = 10,
        RLMESH_DTYPE_FLOAT32 = 11,
        RLMESH_DTYPE_FLOAT64 = 12,
        RLMESH_DTYPE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshSpaceKind : uint
    {
        RLMESH_SPACE_KIND_UNKNOWN = 0,
        RLMESH_SPACE_KIND_BOX = 1,
        RLMESH_SPACE_KIND_DISCRETE = 2,
        RLMESH_SPACE_KIND_MULTI_BINARY = 3,
        RLMESH_SPACE_KIND_MULTI_DISCRETE = 4,
        RLMESH_SPACE_KIND_TEXT = 5,
        RLMESH_SPACE_KIND_DICT = 10,
        RLMESH_SPACE_KIND_TUPLE = 11,
    }

    internal unsafe partial struct RLMeshStr
    {
        [NativeTypeName("const char *")]
        public sbyte* data;

        [NativeTypeName("size_t")]
        public UIntPtr len;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshDevice : uint
    {
        RLMESH_DEVICE_INVALID = 0,
        RLMESH_DEVICE_CPU = 1,
    }

    internal partial struct RLMeshTensor
    {
    }

    internal partial struct RLMeshToDlpackOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte copy;

        [NativeTypeName("_Bool")]
        public byte has_dl_device;

        [NativeTypeName("uint16_t")]
        public ushort _reserved0;

        [NativeTypeName("int32_t")]
        public int dl_device_type;

        [NativeTypeName("int32_t")]
        public int dl_device_id;

        [NativeTypeName("uint32_t")]
        public uint _reserved1;
    }

    internal partial struct RLMeshValue
    {
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshValueKind : uint
    {
        RLMESH_VALUE_INVALID = 0,
        RLMESH_VALUE_NONE = 1,
        RLMESH_VALUE_BOOL = 2,
        RLMESH_VALUE_INT = 3,
        RLMESH_VALUE_FLOAT = 4,
        RLMESH_VALUE_TEXT = 5,
        RLMESH_VALUE_BYTES = 6,
        RLMESH_VALUE_TENSOR = 7,
        RLMESH_VALUE_LIST = 8,
        RLMESH_VALUE_TUPLE = 9,
        RLMESH_VALUE_DICT = 10,
    }

    internal partial struct RLMeshSpaceSpec
    {
    }

    internal partial struct RLMeshEnvContract
    {
    }

    internal unsafe partial struct RLMeshBytes
    {
        [NativeTypeName("uint8_t *")]
        public byte* data;

        [NativeTypeName("size_t")]
        public UIntPtr len;

        [NativeTypeName("size_t")]
        public UIntPtr cap;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshBoundsKind : uint
    {
        RLMESH_BOUNDS_KIND_INVALID = 0,
        RLMESH_BOUNDS_KIND_UNBOUNDED = 1,
        RLMESH_BOUNDS_KIND_UNIFORM = 2,
        RLMESH_BOUNDS_KIND_ELEMENTWISE = 3,
        RLMESH_BOUNDS_KIND_TYPED_UNIFORM = 4,
        RLMESH_BOUNDS_KIND_TYPED_ELEMENTWISE = 5,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshBoundType : uint
    {
        RLMESH_BOUND_TYPE_INVALID = 0,
        RLMESH_BOUND_TYPE_BOOL = 1,
        RLMESH_BOUND_TYPE_INT = 2,
        RLMESH_BOUND_TYPE_UINT = 3,
        RLMESH_BOUND_TYPE_FLOAT = 4,
    }

    internal partial struct RLMeshBound
    {
        public RLMeshBoundType type_;

        [NativeTypeName("__AnonymousRecord_rlmesh_L855_C3")]
        public _value_e__Union value;

        [StructLayout(LayoutKind.Explicit)]
        internal partial struct _value_e__Union
        {
            [FieldOffset(0)]
            [NativeTypeName("_Bool")]
            public byte b;

            [FieldOffset(0)]
            [NativeTypeName("int64_t")]
            public long i;

            [FieldOffset(0)]
            [NativeTypeName("uint64_t")]
            public ulong u;

            [FieldOffset(0)]
            public double f;
        }
    }

    internal partial struct RLMeshSpace
    {
    }

    internal partial struct RLMeshBoxOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        public RLMeshDType dtype;

        [NativeTypeName("uint32_t")]
        public uint _reserved;
    }

    internal partial struct RLMeshDiscreteOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("int64_t")]
        public long start;

        public RLMeshDType dtype;

        [NativeTypeName("uint32_t")]
        public uint _reserved;
    }

    internal partial struct RLMeshEnv
    {
    }

    internal partial struct RLMeshVectorEnv
    {
    }

    internal partial struct RLMeshObsTransform
    {
    }

    internal partial struct RLMeshRotationTransform
    {
    }

    internal partial struct RLMeshModelSpec
    {
    }

    internal partial struct RLMeshEnvTags
    {
    }

    internal partial struct RLMeshAdapter
    {
    }

    internal partial struct RLMeshWrappedPredict
    {
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshImageLayout : uint
    {
        RLMESH_IMAGE_LAYOUT_INVALID = 0,
        RLMESH_IMAGE_LAYOUT_HWC = 1,
        RLMESH_IMAGE_LAYOUT_CHW = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshFitMode : uint
    {
        RLMESH_FIT_MODE_INVALID = 0,
        RLMESH_FIT_MODE_STRETCH = 1,
        RLMESH_FIT_MODE_CROP = 2,
        RLMESH_FIT_MODE_PAD = 3,
        RLMESH_FIT_MODE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshFrame : uint
    {
        RLMESH_FRAME_INVALID = 0,
        RLMESH_FRAME_WORLD = 1,
        RLMESH_FRAME_ROBOT_BASE = 2,
        RLMESH_FRAME_TOOL = 3,
        RLMESH_FRAME_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshReference : uint
    {
        RLMESH_REFERENCE_INVALID = 0,
        RLMESH_REFERENCE_CURRENT = 1,
        RLMESH_REFERENCE_TARGET = 2,
        RLMESH_REFERENCE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshProvenance : uint
    {
        RLMESH_PROVENANCE_INVALID = 0,
        RLMESH_PROVENANCE_SENSED = 1,
        RLMESH_PROVENANCE_ESTIMATED = 2,
        RLMESH_PROVENANCE_PRIVILEGED = 3,
        RLMESH_PROVENANCE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshResample : uint
    {
        RLMESH_RESAMPLE_INVALID = 0,
        RLMESH_RESAMPLE_BILINEAR = 1,
        RLMESH_RESAMPLE_BILINEAR_AA = 2,
        RLMESH_RESAMPLE_BICUBIC_AA = 3,
        RLMESH_RESAMPLE_LANCZOS3_AA = 4,
        RLMESH_RESAMPLE_AREA = 5,
        RLMESH_RESAMPLE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshRotationEncoding : uint
    {
        RLMESH_ROTATION_ENCODING_INVALID = 0,
        RLMESH_ROTATION_ENCODING_QUAT_XYZW = 1,
        RLMESH_ROTATION_ENCODING_QUAT_WXYZ = 2,
        RLMESH_ROTATION_ENCODING_AXIS_ANGLE = 3,
        RLMESH_ROTATION_ENCODING_ROT6D = 4,
        RLMESH_ROTATION_ENCODING_ROT6D_ROWMAJOR = 5,
        RLMESH_ROTATION_ENCODING_EULER_XYZ = 6,
        RLMESH_ROTATION_ENCODING_GRAVITY_XYZ = 7,
        RLMESH_ROTATION_ENCODING_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshStackPad : uint
    {
        RLMESH_STACK_PAD_INVALID = 0,
        RLMESH_STACK_PAD_FIRST = 1,
        RLMESH_STACK_PAD_BLACK = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshCropMode : uint
    {
        RLMESH_CROP_MODE_INVALID = 0,
        RLMESH_CROP_MODE_ZOOM = 1,
        RLMESH_CROP_MODE_SLICE = 2,
        RLMESH_CROP_MODE_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshChannelOrder : uint
    {
        RLMESH_CHANNEL_ORDER_INVALID = 0,
        RLMESH_CHANNEL_ORDER_RGB = 1,
        RLMESH_CHANNEL_ORDER_BGR = 2,
        RLMESH_CHANNEL_ORDER_OTHER = 1000,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshStateContainer : uint
    {
        RLMESH_STATE_CONTAINER_INVALID = 0,
        RLMESH_STATE_CONTAINER_ARRAY = 1,
        RLMESH_STATE_CONTAINER_LIST = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshTextContainer : uint
    {
        RLMESH_TEXT_CONTAINER_INVALID = 0,
        RLMESH_TEXT_CONTAINER_STR = 1,
        RLMESH_TEXT_CONTAINER_LIST = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshStateSource : uint
    {
        RLMESH_STATE_SOURCE_INVALID = 0,
        RLMESH_STATE_SOURCE_OBSERVATION = 1,
        RLMESH_STATE_SOURCE_ACTION = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshAdvisorySeverity : uint
    {
        RLMESH_ADVISORY_SEVERITY_INVALID = 0,
        RLMESH_ADVISORY_SEVERITY_INFO = 1,
        RLMESH_ADVISORY_SEVERITY_CAUTION = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshNormalize : uint
    {
        RLMESH_NORMALIZE_INVALID = 0,
        RLMESH_NORMALIZE_OFF = 1,
        RLMESH_NORMALIZE_UNIT = 2,
        RLMESH_NORMALIZE_RANGE = 3,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshInputNodeKind : uint
    {
        RLMESH_INPUT_NODE_KIND_INVALID = 0,
        RLMESH_INPUT_NODE_KIND_IMAGE = 1,
        RLMESH_INPUT_NODE_KIND_STATE = 2,
        RLMESH_INPUT_NODE_KIND_CONCAT = 3,
        RLMESH_INPUT_NODE_KIND_TEXT = 4,
        RLMESH_INPUT_NODE_KIND_CUSTOM = 5,
        RLMESH_INPUT_NODE_KIND_DICT = 6,
        RLMESH_INPUT_NODE_KIND_TUPLE = 7,
    }

    internal unsafe partial struct RLMeshInputNode
    {
        public RLMeshInputNodeKind kind;

        [NativeTypeName("const void *")]
        public void* leaf;

        [NativeTypeName("const RLMeshStr *")]
        public RLMeshStr* keys;

        [NativeTypeName("const struct RLMeshInputNode *")]
        public RLMeshInputNode* children;

        [NativeTypeName("size_t")]
        public UIntPtr children_len;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshObsNodeKind : uint
    {
        RLMESH_OBS_NODE_KIND_INVALID = 0,
        RLMESH_OBS_NODE_KIND_IMAGE_TAG = 1,
        RLMESH_OBS_NODE_KIND_STATE_TAG = 2,
        RLMESH_OBS_NODE_KIND_TEXT_TAG = 3,
        RLMESH_OBS_NODE_KIND_SPLIT = 4,
        RLMESH_OBS_NODE_KIND_DICT = 5,
        RLMESH_OBS_NODE_KIND_TUPLE = 6,
    }

    internal unsafe partial struct RLMeshObsNode
    {
        public RLMeshObsNodeKind kind;

        [NativeTypeName("const void *")]
        public void* leaf;

        [NativeTypeName("const RLMeshStr *")]
        public RLMeshStr* keys;

        [NativeTypeName("const struct RLMeshObsNode *")]
        public RLMeshObsNode* children;

        [NativeTypeName("size_t")]
        public UIntPtr children_len;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshConcatPartKind : uint
    {
        RLMESH_CONCAT_PART_KIND_INVALID = 0,
        RLMESH_CONCAT_PART_KIND_ROLE = 1,
        RLMESH_CONCAT_PART_KIND_STATE = 2,
        RLMESH_CONCAT_PART_KIND_CONSTANT = 3,
    }

    internal unsafe partial struct RLMeshConcatPart
    {
        public RLMeshConcatPartKind kind;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("const void *")]
        public void* leaf;
    }

    internal unsafe partial struct RLMeshEncodingArm
    {
        [NativeTypeName("const RLMeshRotationTransform *")]
        public RLMeshRotationTransform* transform;

        [NativeTypeName("const char *")]
        public sbyte* entrypoint;
    }

    internal unsafe partial struct RLMeshRotation
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        public RLMeshRotationEncoding encoding;

        [NativeTypeName("uint32_t")]
        public uint _reserved;

        [NativeTypeName("const double *")]
        public double* value;

        [NativeTypeName("size_t")]
        public UIntPtr value_len;
    }

    internal unsafe partial struct RLMeshCustomEncoding
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        public RLMeshRotationEncoding @base;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved0[4];

        public RLMeshEncodingArm from_base;

        public RLMeshEncodingArm to_base;

        [NativeTypeName("const char *")]
        public sbyte* name;
    }

    internal unsafe partial struct RLMeshImage
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("_Bool")]
        public byte has_height;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        [NativeTypeName("uint32_t")]
        public uint height;

        [NativeTypeName("_Bool")]
        public byte has_width;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved1[3];

        [NativeTypeName("uint32_t")]
        public uint width;

        public RLMeshImageLayout layout;

        [NativeTypeName("_Bool")]
        public byte has_channels;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved2[3];

        [NativeTypeName("uint32_t")]
        public uint channels;

        public RLMeshDType dtype;

        [NativeTypeName("const char *")]
        public sbyte* dtype_other;

        public RLMeshNormalize normalize;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved3[4];

        [NativeTypeName("double[2]")]
        public fixed double normalize_range[2];

        [NativeTypeName("uint32_t")]
        public uint lead_dims;

        [NativeTypeName("_Bool")]
        public byte upside_down;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved4[3];

        public RLMeshResample resample;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved5[4];

        [NativeTypeName("const char *")]
        public sbyte* resample_other;

        [NativeTypeName("_Bool")]
        public byte allow_upscale;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved6[7];

        [NativeTypeName("const RLMeshFitMode *")]
        public RLMeshFitMode* fit;

        [NativeTypeName("size_t")]
        public UIntPtr fit_len;

        [NativeTypeName("const char *const *")]
        public sbyte** fit_other;

        [NativeTypeName("_Bool")]
        public byte optional;

        [NativeTypeName("_Bool")]
        public byte has_fill;

        [NativeTypeName("uint8_t")]
        public byte fill;

        [NativeTypeName("uint8_t[1]")]
        public fixed byte _reserved7[1];

        [NativeTypeName("uint32_t")]
        public uint stack;

        [NativeTypeName("_Bool")]
        public byte has_crop;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved8[7];

        public double crop;

        [NativeTypeName("_Bool")]
        public byte has_crop_area;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved9[7];

        public double crop_area;

        public RLMeshCropMode crop_mode;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved10[4];

        [NativeTypeName("const char *")]
        public sbyte* crop_mode_other;

        [NativeTypeName("_Bool")]
        public byte has_jpeg_quality;

        [NativeTypeName("uint8_t")]
        public byte jpeg_quality;

        [NativeTypeName("uint8_t[2]")]
        public fixed byte _reserved11[2];

        public RLMeshChannelOrder channel_order;

        [NativeTypeName("const char *")]
        public sbyte* channel_order_other;

        [NativeTypeName("_Bool")]
        public byte has_render;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved12[3];

        [NativeTypeName("uint32_t[2]")]
        public fixed uint render[2];

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved13[4];

        [NativeTypeName("const int32_t *")]
        public int* offsets;

        [NativeTypeName("size_t")]
        public UIntPtr offsets_len;

        public RLMeshStackPad stack_pad;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved14[4];

        [NativeTypeName("const char *")]
        public sbyte* part;

        [NativeTypeName("_Bool")]
        public byte has_stride;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved15[3];

        [NativeTypeName("uint32_t")]
        public uint stride;
    }

    internal unsafe partial struct RLMeshState
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("const RLMeshRotationEncoding *")]
        public RLMeshRotationEncoding* encoding;

        [NativeTypeName("size_t")]
        public UIntPtr encoding_len;

        [NativeTypeName("const char *const *")]
        public sbyte** encoding_other;

        [NativeTypeName("const RLMeshCustomEncoding *")]
        public RLMeshCustomEncoding* custom_encoding;

        [NativeTypeName("_Bool")]
        public byte has_dim;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        [NativeTypeName("uint32_t")]
        public uint dim;

        [NativeTypeName("_Bool")]
        public byte has_index;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved1[3];

        [NativeTypeName("uint32_t")]
        public uint index;

        [NativeTypeName("_Bool")]
        public byte optional;

        [NativeTypeName("_Bool")]
        public byte has_range;

        [NativeTypeName("uint8_t[6]")]
        public fixed byte _reserved2[6];

        [NativeTypeName("double[2]")]
        public fixed double range[2];

        [NativeTypeName("_Bool")]
        public byte has_pad_to;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved3[3];

        [NativeTypeName("uint32_t")]
        public uint pad_to;

        public RLMeshDType dtype;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved4[4];

        [NativeTypeName("const char *")]
        public sbyte* dtype_other;

        [NativeTypeName("const int64_t *")]
        public long* reshape;

        [NativeTypeName("size_t")]
        public UIntPtr reshape_len;

        public RLMeshStateContainer container;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved5[4];

        public double fill;

        [NativeTypeName("const RLMeshRotation *")]
        public RLMeshRotation* post_rotate;

        [NativeTypeName("_Bool")]
        public byte has_scale;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved6[7];

        public double scale;

        [NativeTypeName("const double *")]
        public double* axis_scale;

        [NativeTypeName("size_t")]
        public UIntPtr axis_scale_len;

        [NativeTypeName("_Bool")]
        public byte has_offset;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved7[7];

        public double offset;

        [NativeTypeName("const double *")]
        public double* axis_offset;

        [NativeTypeName("size_t")]
        public UIntPtr axis_offset_len;

        [NativeTypeName("_Bool")]
        public byte has_frame;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved8[3];

        public RLMeshFrame frame;

        [NativeTypeName("const char *")]
        public sbyte* frame_other;

        [NativeTypeName("const RLMeshProvenance *")]
        public RLMeshProvenance* provenance;

        [NativeTypeName("size_t")]
        public UIntPtr provenance_len;

        [NativeTypeName("const char *const *")]
        public sbyte** provenance_other;

        [NativeTypeName("const char *")]
        public sbyte* part;

        [NativeTypeName("const char *const *")]
        public sbyte** labels;

        [NativeTypeName("size_t")]
        public UIntPtr labels_len;

        public RLMeshStateSource source;

        [NativeTypeName("_Bool")]
        public byte has_clip;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved9[3];

        [NativeTypeName("double[2]")]
        public fixed double clip[2];

        [NativeTypeName("const char *")]
        public sbyte* polarity;
    }

    internal unsafe partial struct RLMeshConcat
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshConcatPart *")]
        public RLMeshConcatPart* parts;

        [NativeTypeName("size_t")]
        public UIntPtr parts_len;

        [NativeTypeName("_Bool")]
        public byte has_pad_to;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        [NativeTypeName("uint32_t")]
        public uint pad_to;

        public RLMeshDType dtype;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved1[4];

        [NativeTypeName("const char *")]
        public sbyte* dtype_other;

        [NativeTypeName("const int64_t *")]
        public long* reshape;

        [NativeTypeName("size_t")]
        public UIntPtr reshape_len;

        public RLMeshStateContainer container;

        [NativeTypeName("_Bool")]
        public byte has_clip;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved2[3];

        [NativeTypeName("double[2]")]
        public fixed double clip[2];
    }

    internal unsafe partial struct RLMeshConstant
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("uint32_t")]
        public uint dim;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved0[4];

        public double fill;
    }

    internal unsafe partial struct RLMeshAdaptersText
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        public RLMeshTextContainer container;

        [NativeTypeName("_Bool")]
        public byte has_fill;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        public RLMeshStr fill;
    }

    internal unsafe partial struct RLMeshCustom
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshObsTransform *")]
        public RLMeshObsTransform* transform;

        [NativeTypeName("const char *")]
        public sbyte* entrypoint;
    }

    internal unsafe partial struct RLMeshActuator
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("uint32_t")]
        public uint dim;

        [NativeTypeName("_Bool")]
        public byte has_encoding;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        public RLMeshRotationEncoding encoding;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved1[4];

        [NativeTypeName("const RLMeshCustomEncoding *")]
        public RLMeshCustomEncoding* custom_encoding;

        [NativeTypeName("_Bool")]
        public byte has_range;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved2[7];

        [NativeTypeName("double[2]")]
        public fixed double range[2];

        [NativeTypeName("_Bool")]
        public byte binary;

        [NativeTypeName("_Bool")]
        public byte has_scale;

        [NativeTypeName("uint8_t[6]")]
        public fixed byte _reserved3[6];

        public double scale;

        [NativeTypeName("const double *")]
        public double* axis_scale;

        [NativeTypeName("size_t")]
        public UIntPtr axis_scale_len;

        [NativeTypeName("_Bool")]
        public byte invert;

        [NativeTypeName("_Bool")]
        public byte has_threshold;

        [NativeTypeName("uint8_t[6]")]
        public fixed byte _reserved4[6];

        public double threshold;

        [NativeTypeName("_Bool")]
        public byte clip;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved5[7];

        public double fill;

        [NativeTypeName("_Bool")]
        public byte optional;

        [NativeTypeName("_Bool")]
        public byte has_frame;

        [NativeTypeName("uint8_t[2]")]
        public fixed byte _reserved6[2];

        public RLMeshFrame frame;

        [NativeTypeName("const char *")]
        public sbyte* frame_other;

        [NativeTypeName("_Bool")]
        public byte has_reference;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved7[3];

        public RLMeshReference reference;

        [NativeTypeName("const char *")]
        public sbyte* reference_other;

        [NativeTypeName("const char *")]
        public sbyte* part;

        [NativeTypeName("_Bool")]
        public byte has_offset;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved8[7];

        public double offset;

        [NativeTypeName("const double *")]
        public double* axis_offset;

        [NativeTypeName("size_t")]
        public UIntPtr axis_offset_len;

        [NativeTypeName("const char *const *")]
        public sbyte** labels;

        [NativeTypeName("size_t")]
        public UIntPtr labels_len;

        [NativeTypeName("const char *")]
        public sbyte* polarity;

        [NativeTypeName("_Bool")]
        public byte has_unit;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved9[7];

        public double unit;
    }

    internal unsafe partial struct RLMeshAction
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshActuator *const *")]
        public RLMeshActuator** components;

        [NativeTypeName("size_t")]
        public UIntPtr components_len;

        [NativeTypeName("_Bool")]
        public byte has_clip;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved0[7];

        [NativeTypeName("double[2]")]
        public fixed double clip[2];
    }

    internal unsafe partial struct RLMeshImageTag
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        public RLMeshImageLayout layout;

        [NativeTypeName("_Bool")]
        public byte upside_down;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved0[3];

        [NativeTypeName("const char *")]
        public sbyte* part;
    }

    internal unsafe partial struct RLMeshStateTag
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("const RLMeshRotationEncoding *")]
        public RLMeshRotationEncoding* encoding;

        [NativeTypeName("size_t")]
        public UIntPtr encoding_len;

        [NativeTypeName("const char *const *")]
        public sbyte** encoding_other;

        [NativeTypeName("_Bool")]
        public byte has_range;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved0[7];

        [NativeTypeName("double[2]")]
        public fixed double range[2];

        [NativeTypeName("_Bool")]
        public byte has_frame;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved1[3];

        public RLMeshFrame frame;

        [NativeTypeName("const char *")]
        public sbyte* frame_other;

        [NativeTypeName("_Bool")]
        public byte has_provenance;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved2[3];

        public RLMeshProvenance provenance;

        [NativeTypeName("const char *")]
        public sbyte* provenance_other;

        [NativeTypeName("const char *")]
        public sbyte* part;

        [NativeTypeName("const char *const *")]
        public sbyte** labels;

        [NativeTypeName("size_t")]
        public UIntPtr labels_len;

        [NativeTypeName("const char *")]
        public sbyte* polarity;
    }

    internal unsafe partial struct RLMeshTextTag
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;
    }

    internal unsafe partial struct RLMeshField
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* role;

        [NativeTypeName("uint32_t")]
        public uint dim;

        [NativeTypeName("uint8_t[4]")]
        public fixed byte _reserved0[4];

        [NativeTypeName("const RLMeshRotationEncoding *")]
        public RLMeshRotationEncoding* encoding;

        [NativeTypeName("size_t")]
        public UIntPtr encoding_len;

        [NativeTypeName("const char *const *")]
        public sbyte** encoding_other;

        [NativeTypeName("_Bool")]
        public byte has_range;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved1[7];

        [NativeTypeName("double[2]")]
        public fixed double range[2];

        [NativeTypeName("_Bool")]
        public byte has_frame;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved2[3];

        public RLMeshFrame frame;

        [NativeTypeName("const char *")]
        public sbyte* frame_other;

        [NativeTypeName("_Bool")]
        public byte has_provenance;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved3[3];

        public RLMeshProvenance provenance;

        [NativeTypeName("const char *")]
        public sbyte* provenance_other;

        [NativeTypeName("const char *")]
        public sbyte* part;

        [NativeTypeName("const char *const *")]
        public sbyte** labels;

        [NativeTypeName("size_t")]
        public UIntPtr labels_len;

        [NativeTypeName("const char *")]
        public sbyte* polarity;
    }

    internal unsafe partial struct RLMeshSplit
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshField *const *")]
        public RLMeshField** fields;

        [NativeTypeName("size_t")]
        public UIntPtr fields_len;
    }

    internal unsafe partial struct RLMeshPreviousOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* part;

        public double fill;
    }

    internal unsafe partial struct RLMeshObservationRoles
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *const *")]
        public sbyte** images;

        [NativeTypeName("size_t")]
        public UIntPtr images_len;

        [NativeTypeName("const char *const *")]
        public sbyte** states;

        [NativeTypeName("size_t")]
        public UIntPtr states_len;

        [NativeTypeName("const char *const *")]
        public sbyte** texts;

        [NativeTypeName("size_t")]
        public UIntPtr texts_len;
    }

    internal unsafe partial struct RLMeshAdvisory
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        public RLMeshAdvisorySeverity severity;

        [NativeTypeName("uint32_t")]
        public uint _reserved;

        [NativeTypeName("const char *")]
        public sbyte* message;
    }

    internal partial struct RLMeshTagOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte validate;

        [NativeTypeName("uint8_t")]
        public byte _reserved0;

        [NativeTypeName("uint16_t")]
        public ushort _reserved1;

        [NativeTypeName("uint32_t")]
        public uint _reserved2;
    }

    internal partial struct RLMeshResolveOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte trust_entrypoints;

        [NativeTypeName("_Bool")]
        public byte check_inverse;

        [NativeTypeName("uint16_t")]
        public ushort _reserved0;

        [NativeTypeName("uint32_t")]
        public uint _reserved1;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate RLMeshStatus RLMeshObsTransformFn(void* user_data, [NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshValue** @out);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate RLMeshStatus RLMeshRotationTransformFn(void* user_data, [NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshValue** @out);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate RLMeshStatus RLMeshPredictFn(void* user_data, [NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshValue** @out);

    internal partial struct RLMeshAdapterVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshValue **)")]
        public IntPtr transform_obs;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshValue **)")]
        public IntPtr transform_action;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr reset;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *)")]
        public IntPtr observe;

        [NativeTypeName("RLMeshStatus (*)(void *, const char **)")]
        public IntPtr explain;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    internal partial struct RLMeshEnvServer
    {
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshTransport : uint
    {
        RLMESH_TRANSPORT_INVALID = 0,
        RLMESH_TRANSPORT_TCP = 1,
        RLMESH_TRANSPORT_UNIX = 2,
    }

    internal unsafe partial struct RLMeshServeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte allow_remote_shutdown;

        [NativeTypeName("_Bool")]
        public byte has_idle_timeout_seconds;

        [NativeTypeName("_Bool")]
        public byte has_drain_timeout_seconds;

        [NativeTypeName("_Bool")]
        public byte has_close_timeout_seconds;

        [NativeTypeName("_Bool")]
        public byte compress_responses;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved[3];

        public double idle_timeout_seconds;

        public double drain_timeout_seconds;

        public double close_timeout_seconds;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;
    }

    internal unsafe partial struct RLMeshExtEnvServerOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* token;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* peer_info_extra;

        [NativeTypeName("const char *")]
        public sbyte* host_runtime_json;

        [NativeTypeName("_Bool")]
        public byte publish_describe;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshEnvServerOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* address;

        [NativeTypeName("const char *")]
        public sbyte* host;

        [NativeTypeName("const char *")]
        public sbyte* path;

        [NativeTypeName("const RLMeshServeOptions *")]
        public RLMeshServeOptions* options;

        [NativeTypeName("const RLMeshEnvTags *")]
        public RLMeshEnvTags* tags;

        [NativeTypeName("const RLMeshExtEnvServerOptions *")]
        public RLMeshExtEnvServerOptions* ext;

        [NativeTypeName("uint16_t")]
        public ushort port;

        [NativeTypeName("_Bool")]
        public byte has_port;

        [NativeTypeName("_Bool")]
        public byte has_transport;

        public RLMeshTransport transport;

        [NativeTypeName("_Bool")]
        public byte close_env_on_shutdown;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshEnvAttributes
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshSpace *")]
        public RLMeshSpace* observation_space;

        [NativeTypeName("const RLMeshSpace *")]
        public RLMeshSpace* action_space;

        [NativeTypeName("const char *")]
        public sbyte* render_mode;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* metadata;

        [NativeTypeName("const char *")]
        public sbyte* id;
    }

    internal unsafe partial struct RLMeshVectorEnvAttributes
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("uint64_t")]
        public ulong num_envs;

        [NativeTypeName("const RLMeshSpace *")]
        public RLMeshSpace* single_observation_space;

        [NativeTypeName("const RLMeshSpace *")]
        public RLMeshSpace* single_action_space;

        [NativeTypeName("const char *")]
        public sbyte* render_mode;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* metadata;

        [NativeTypeName("const char *")]
        public sbyte* id;
    }

    internal partial struct RLMeshEnvVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, const int64_t *, const RLMeshValue *, RLMeshValue **, RLMeshValue **)")]
        public IntPtr reset;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshValue **, double *, _Bool *, _Bool *, RLMeshValue **)")]
        public IntPtr step;

        [NativeTypeName("RLMeshStatus (*)(void *, RLMeshValue **)")]
        public IntPtr render;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr close;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    internal partial struct RLMeshVectorEnvVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, const int64_t *, size_t, const RLMeshValue *, RLMeshValue **, RLMeshValue **)")]
        public IntPtr reset;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshValue **, double *, _Bool *, _Bool *, RLMeshValue **)")]
        public IntPtr step;

        [NativeTypeName("RLMeshStatus (*)(void *, RLMeshValue **)")]
        public IntPtr render;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr close;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    internal partial struct RLMeshEnvFactory
    {
    }

    internal partial struct RLMeshVariant
    {
    }

    internal partial struct RLMeshVariantSink
    {
    }

    internal partial struct RLMeshVariantList
    {
    }

    internal partial struct RLMeshTagParamList
    {
    }

    internal partial struct RLMeshExtCancelToken
    {
    }

    internal unsafe partial struct RLMeshExtEnvFactoryServeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* host_runtime_json;
    }

    internal unsafe partial struct RLMeshEnvFactoryServeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("int64_t")]
        public long num_envs;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* make_kwargs;

        public RLMeshExtCancelToken* cancel;

        [NativeTypeName("uint64_t")]
        public ulong _reserved;

        [NativeTypeName("const RLMeshExtEnvFactoryServeOptions *")]
        public RLMeshExtEnvFactoryServeOptions* ext;
    }

    internal unsafe partial struct RLMeshTagParam
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* name;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* default_;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* choices;
    }

    internal unsafe partial struct RLMeshEnvFactoryAttributes
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshEnvTags *")]
        public RLMeshEnvTags* tags;

        [NativeTypeName("const RLMeshTagParam *const *")]
        public RLMeshTagParam** tag_params;

        [NativeTypeName("size_t")]
        public UIntPtr tag_params_len;

        [NativeTypeName("const char *const *")]
        public sbyte** reset_options;

        [NativeTypeName("size_t")]
        public UIntPtr reset_options_len;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;
    }

    internal partial struct RLMeshEnvFactoryVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr prepare;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshEnv **, RLMeshVectorEnv **)")]
        public IntPtr make;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshEnvTags **)")]
        public IntPtr tags_for;

        [NativeTypeName("RLMeshStatus (*)(void *, RLMeshVariantSink *)")]
        public IntPtr enumerate_variants;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr close;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    internal unsafe partial struct RLMeshVariantOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* metadata;
    }

    internal partial struct RLMeshModel
    {
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshSpecKind : uint
    {
        RLMESH_SPEC_KIND_INVALID = 0,
        RLMESH_SPEC_KIND_NONE = 1,
        RLMESH_SPEC_KIND_MODEL_SPEC = 2,
        RLMESH_SPEC_KIND_NO_ADAPTER = 3,
    }

    internal unsafe partial struct RLMeshSpecArg
    {
        public RLMeshSpecKind kind;

        [NativeTypeName("const RLMeshModelSpec *")]
        public RLMeshModelSpec* spec;
    }

    internal partial struct RLMeshEpisodeState
    {
    }

    internal unsafe partial struct RLMeshPredictContext
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* episode_id;

        [NativeTypeName("_Bool")]
        public byte has_episode_seed;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved0[7];

        [NativeTypeName("int64_t")]
        public long episode_seed;

        [NativeTypeName("uint64_t")]
        public ulong predict_index;

        [NativeTypeName("_Bool")]
        public byte has_predict_seed;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved1[7];

        [NativeTypeName("int64_t")]
        public long predict_seed;

        public RLMeshEpisodeState* state;
    }

    internal unsafe partial struct RLMeshLoadKey
    {
        [NativeTypeName("const char *")]
        public sbyte* name;

        [NativeTypeName("_Bool")]
        public byte required;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshModelVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, const RLMeshPredictContext *, RLMeshValue **)")]
        public IntPtr predict;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, uint32_t, const RLMeshPredictContext *, RLMeshValue **)")]
        public IntPtr predict_chunk;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *const *, size_t, const RLMeshPredictContext *const *, RLMeshValue **)")]
        public IntPtr predict_batch;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *const *, size_t, uint32_t, const RLMeshPredictContext *const *, RLMeshValue **)")]
        public IntPtr predict_chunk_batch;

        [NativeTypeName("RLMeshStatus (*)(void *, const char *)")]
        public IntPtr on_episode_end;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr close;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *)")]
        public IntPtr load;

        [NativeTypeName("RLMeshStatus (*)(void *, _Bool *, uint32_t *)")]
        public IntPtr loaded_native_chunk;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;

        [NativeTypeName("uint32_t")]
        public uint context_corners;

        [NativeTypeName("uint32_t")]
        public uint _reserved0;

        public RLMeshSpecArg spec;

        [NativeTypeName("_Bool")]
        public byte allow_fusion;

        [NativeTypeName("_Bool")]
        public byte has_native_chunk;

        [NativeTypeName("uint8_t[2]")]
        public fixed byte _reserved1[2];

        [NativeTypeName("uint32_t")]
        public uint native_chunk;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("const RLMeshLoadKey *")]
        public RLMeshLoadKey* load_keys;

        [NativeTypeName("size_t")]
        public UIntPtr load_keys_len;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, RLMeshValue **)")]
        public IntPtr reset_options;
    }

    internal unsafe partial struct RLMeshModelOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        public RLMeshSpecArg spec;

        [NativeTypeName("RLMeshStatus (*)(void *, const char *)")]
        public IntPtr on_episode_end;

        [NativeTypeName("RLMeshStatus (*)(void *)")]
        public IntPtr on_close;

        public void* hook_user_data;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_hook_user_data;

        [NativeTypeName("_Bool")]
        public byte trust_entrypoints;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshExtModelServeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* token;

        [NativeTypeName("size_t")]
        public UIntPtr predict_concurrency;

        [NativeTypeName("_Bool")]
        public byte has_predict_concurrency;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];

        [NativeTypeName("_Bool")]
        public byte no_signal_handlers;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved1[7];
    }

    internal unsafe partial struct RLMeshModelServeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshServeOptions *")]
        public RLMeshServeOptions* options;

        public RLMeshExtCancelToken* cancel;

        [NativeTypeName("const RLMeshExtModelServeOptions *")]
        public RLMeshExtModelServeOptions* ext;

        [NativeTypeName("const char *")]
        public sbyte* name;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshDescribeKind : uint
    {
        RLMESH_DESCRIBE_KIND_INVALID = 0,
        RLMESH_DESCRIBE_KIND_ENV = 1,
        RLMESH_DESCRIBE_KIND_MODEL = 2,
    }

    internal unsafe partial struct RLMeshDescribeOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* generated_at;

        public RLMeshDescribeKind kind;

        [NativeTypeName("_Bool")]
        public byte has_kind;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved[3];
    }

    internal unsafe partial struct RLMeshDescribeJsonOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* generated_at;

        [NativeTypeName("const RLMeshEnv *const *")]
        public RLMeshEnv** served_env;

        [NativeTypeName("size_t")]
        public UIntPtr served_env_len;

        [NativeTypeName("const RLMeshVectorEnv *")]
        public RLMeshVectorEnv* served_vector_env;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        public RLMeshDescribeKind kind;

        [NativeTypeName("_Bool")]
        public byte has_kind;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved[3];
    }

    internal partial struct RLMeshRunResult
    {
    }

    internal partial struct RLMeshTelemetryRow
    {
    }

    internal partial struct RLMeshRunHooks
    {
    }

    internal partial struct RLMeshRunContext
    {
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshModelTargetKind : uint
    {
        RLMESH_MODEL_TARGET_KIND_INVALID = 0,
        RLMESH_MODEL_TARGET_KIND_MODEL = 1,
        RLMESH_MODEL_TARGET_KIND_RANDOM_SAMPLE = 3,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshEnvTargetKind : uint
    {
        RLMESH_ENV_TARGET_KIND_INVALID = 0,
        RLMESH_ENV_TARGET_KIND_ADDRESS = 1,
        RLMESH_ENV_TARGET_KIND_ENV_SERVER = 2,
        RLMESH_ENV_TARGET_KIND_REMOTE_ENV = 3,
        RLMESH_ENV_TARGET_KIND_REMOTE_VECTOR_ENV = 4,
        RLMESH_ENV_TARGET_KIND_ENV = 5,
        RLMESH_ENV_TARGET_KIND_VECTOR_ENV = 6,
        RLMESH_ENV_TARGET_KIND_ENV_FACTORY = 7,
    }

    internal unsafe partial struct RLMeshModelTarget
    {
        public RLMeshModelTargetKind kind;

        [NativeTypeName("const void *")]
        public void* handle;
    }

    internal unsafe partial struct RLMeshEnvTarget
    {
        public RLMeshEnvTargetKind kind;

        [NativeTypeName("const void *")]
        public void* handle;

        [NativeTypeName("const char *")]
        public sbyte* address;
    }

    internal partial struct RLMeshEpisodeResult
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("int64_t")]
        public long index;

        [NativeTypeName("_Bool")]
        public byte has_seed;

        [NativeTypeName("int64_t")]
        public long seed;

        [NativeTypeName("int64_t")]
        public long steps;

        public double reward;

        [NativeTypeName("_Bool")]
        public byte terminated;

        [NativeTypeName("_Bool")]
        public byte truncated;

        [NativeTypeName("_Bool")]
        public byte has_success;

        [NativeTypeName("_Bool")]
        public byte success;

        public double duration_s;

        [NativeTypeName("_Bool")]
        public byte has_predict_ms;

        public double predict_ms;

        [NativeTypeName("_Bool")]
        public byte has_step_ms;

        public double step_ms;

        [NativeTypeName("_Bool")]
        public byte has_trial;

        [NativeTypeName("int64_t")]
        public long trial;
    }

    internal unsafe partial struct RLMeshRunResultOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshEpisodeResult *const *")]
        public RLMeshEpisodeResult** episodes;

        [NativeTypeName("size_t")]
        public UIntPtr episodes_len;

        [NativeTypeName("const RLMeshTelemetryRow *const *")]
        public RLMeshTelemetryRow** telemetry;

        [NativeTypeName("size_t")]
        public UIntPtr telemetry_len;

        [NativeTypeName("const RLMeshAdvisory *const *")]
        public RLMeshAdvisory** advisories;

        [NativeTypeName("size_t")]
        public UIntPtr advisories_len;
    }

    internal unsafe partial struct RLMeshRunResultSaveOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* model;

        [NativeTypeName("const char *")]
        public sbyte* env;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* config;
    }

    internal partial struct RLMeshReadItem
    {
    }

    internal partial struct RLMeshReader
    {
    }

    internal partial struct RLMeshFrameSources
    {
    }

    internal partial struct RLMeshRunContextVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, RLMeshEnvContract **)")]
        public IntPtr contract;

        [NativeTypeName("RLMeshStatus (*)(void *, uint64_t *)")]
        public IntPtr num_envs;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshValue *, const RLMeshReadItem *, RLMeshValue **)")]
        public IntPtr read;

        [NativeTypeName("RLMeshStatus (*)(void *, RLMeshFrameSources **)")]
        public IntPtr frame_sources;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    internal unsafe partial struct RLMeshStepEvent
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("int64_t")]
        public long episode;

        [NativeTypeName("_Bool")]
        public byte has_seed;

        [NativeTypeName("int64_t")]
        public long seed;

        [NativeTypeName("int64_t")]
        public long step;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* observation;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* action;

        public double reward;

        [NativeTypeName("_Bool")]
        public byte terminated;

        [NativeTypeName("_Bool")]
        public byte truncated;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* info;

        public double predict_ms;

        public double step_ms;

        [NativeTypeName("const RLMeshRunContext *")]
        public RLMeshRunContext* context;
    }

    internal partial struct RLMeshRunHooksVtable
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshRunContext *)")]
        public IntPtr on_run_start;

        [NativeTypeName("RLMeshStatus (*)(void *, int64_t, const int64_t *)")]
        public IntPtr on_episode_start;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshStepEvent *)")]
        public IntPtr on_step;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshEpisodeResult *)")]
        public IntPtr on_episode_end;

        [NativeTypeName("RLMeshStatus (*)(void *, const RLMeshRunResult *)")]
        public IntPtr on_run_end;

        [NativeTypeName("void (*)(void *)")]
        public IntPtr free_user_data;
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshViewBackend : uint
    {
        RLMESH_VIEW_BACKEND_INVALID = 0,
        RLMESH_VIEW_BACKEND_TERMINAL = 1,
        RLMESH_VIEW_BACKEND_HTTP = 2,
        RLMESH_VIEW_BACKEND_BOTH = 3,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshViewFormat : uint
    {
        RLMESH_VIEW_FORMAT_INVALID = 0,
        RLMESH_VIEW_FORMAT_JPEG = 1,
        RLMESH_VIEW_FORMAT_PNG = 2,
    }

    [NativeTypeName("unsigned int")]
    internal enum RLMeshViewHold : uint
    {
        RLMESH_VIEW_HOLD_INVALID = 0,
        RLMESH_VIEW_HOLD_OFF = 1,
        RLMESH_VIEW_HOLD_UNTIL_QUIT = 2,
        RLMESH_VIEW_HOLD_SECONDS = 3,
    }

    internal unsafe partial struct RLMeshView
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* source;

        [NativeTypeName("uint64_t")]
        public ulong fps;

        public double hold_seconds;

        public double step_hz;

        public RLMeshViewBackend backend;

        public RLMeshViewFormat format;

        public RLMeshViewHold hold;

        [NativeTypeName("uint16_t")]
        public ushort port;

        [NativeTypeName("uint8_t")]
        public byte quality;

        [NativeTypeName("_Bool")]
        public byte has_step_hz;
    }

    internal unsafe partial struct RLMeshRunOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const int64_t *")]
        public long* seeds;

        [NativeTypeName("size_t")]
        public UIntPtr seeds_len;

        public RLMeshRunHooks* hooks;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("const RLMeshExtCancelToken *")]
        public RLMeshExtCancelToken* cancel;

        [NativeTypeName("uint64_t")]
        public ulong episodes;

        [NativeTypeName("uint64_t")]
        public ulong max_episode_steps;

        public double max_episode_seconds;

        [NativeTypeName("uint64_t")]
        public ulong prefetch_lead;

        [NativeTypeName("uint64_t")]
        public ulong trial_index_base;

        [NativeTypeName("uint32_t")]
        public uint execution_horizon;

        [NativeTypeName("_Bool")]
        public byte has_episodes;

        [NativeTypeName("_Bool")]
        public byte has_max_episode_steps;

        [NativeTypeName("_Bool")]
        public byte has_max_episode_seconds;

        [NativeTypeName("_Bool")]
        public byte close_env;

        [NativeTypeName("_Bool")]
        public byte has_trust_entrypoints;

        [NativeTypeName("_Bool")]
        public byte trust_entrypoints;

        [NativeTypeName("uint8_t[6]")]
        public fixed byte _reserved[6];

        [NativeTypeName("const char *")]
        public sbyte* instruction;

        [NativeTypeName("const RLMeshView *")]
        public RLMeshView* view;
    }

    internal partial struct RLMeshBinding
    {
    }

    internal unsafe partial struct RLMeshBindOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("uint32_t")]
        public uint execution_horizon;

        [NativeTypeName("_Bool")]
        public byte close_env;

        [NativeTypeName("uint8_t[3]")]
        public fixed byte _reserved[3];

        public RLMeshRunHooks* hooks;
    }

    internal partial struct RLMeshRemoteEnv
    {
    }

    internal partial struct RLMeshRemoteVectorEnv
    {
    }

    internal unsafe partial struct RLMeshRemoteEnvOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* address;

        [NativeTypeName("const char *")]
        public sbyte* host;

        [NativeTypeName("_Bool")]
        public byte has_port;

        [NativeTypeName("uint16_t")]
        public ushort port;

        [NativeTypeName("const char *")]
        public sbyte* path;

        [NativeTypeName("_Bool")]
        public byte has_transport;

        public RLMeshTransport transport;

        [NativeTypeName("_Bool")]
        public byte has_connect_timeout_seconds;

        public double connect_timeout_seconds;

        [NativeTypeName("_Bool")]
        public byte has_request_timeout_seconds;

        public double request_timeout_seconds;

        [NativeTypeName("const char *")]
        public sbyte* workflow_edition;

        [NativeTypeName("uint64_t")]
        public ulong _reserved;

        [NativeTypeName("const char *")]
        public sbyte* api_key;

        [NativeTypeName("const char *")]
        public sbyte* tls_ca_file;
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    internal unsafe delegate RLMeshStatus RLMeshRenderFn(void* user_data, RLMeshValue** @out);

    internal partial struct RLMeshRecorder
    {
    }

    internal partial struct RLMeshMediaRef
    {
    }

    internal partial struct RLMeshEpisodeRecord
    {
    }

    internal partial struct RLMeshWorkloadRecord
    {
    }

    internal unsafe partial struct RLMeshRecorderOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* result_set_id;

        [NativeTypeName("uint64_t")]
        public ulong fps;

        [NativeTypeName("uint8_t")]
        public byte quality;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshRecorderAddOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* task;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* config;

        [NativeTypeName("_Bool")]
        public byte included_in_metrics;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshRecorderCaptureOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* task;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* config;

        [NativeTypeName("const char *const *")]
        public sbyte** cameras;

        [NativeTypeName("size_t")]
        public UIntPtr cameras_len;

        [NativeTypeName("const RLMeshRunContext *")]
        public RLMeshRunContext* binding;

        [NativeTypeName("const char *const *")]
        public sbyte** video_info_keys;

        [NativeTypeName("size_t")]
        public UIntPtr video_info_keys_len;

        [NativeTypeName("_Bool")]
        public byte included_in_metrics;

        [NativeTypeName("uint8_t[7]")]
        public fixed byte _reserved[7];
    }

    internal unsafe partial struct RLMeshRecorderToDictOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* recorded_at;
    }

    internal unsafe partial struct RLMeshRecorderExportOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const char *")]
        public sbyte* recorded_at;

        [NativeTypeName("_Bool")]
        public byte has_archive;

        [NativeTypeName("_Bool")]
        public byte archive;

        [NativeTypeName("uint8_t[6]")]
        public fixed byte _reserved[6];
    }

    internal partial struct RLMeshMediaRefOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte has_frame_count;

        [NativeTypeName("int64_t")]
        public long frame_count;

        [NativeTypeName("_Bool")]
        public byte has_width;

        [NativeTypeName("int64_t")]
        public long width;

        [NativeTypeName("_Bool")]
        public byte has_height;

        [NativeTypeName("int64_t")]
        public long height;

        [NativeTypeName("_Bool")]
        public byte has_fps;

        public double fps;
    }

    internal unsafe partial struct RLMeshEpisodeRecordOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("_Bool")]
        public byte has_success;

        [NativeTypeName("_Bool")]
        public byte success;

        public double duration_s;

        [NativeTypeName("_Bool")]
        public byte included_in_metrics;

        [NativeTypeName("const RLMeshMediaRef *const *")]
        public RLMeshMediaRef** media;

        [NativeTypeName("size_t")]
        public UIntPtr media_len;
    }

    internal unsafe partial struct RLMeshWorkloadRecordOptions
    {
        [NativeTypeName("size_t")]
        public UIntPtr struct_size;

        [NativeTypeName("const RLMeshValue *")]
        public RLMeshValue* config;

        [NativeTypeName("const RLMeshEpisodeRecord *const *")]
        public RLMeshEpisodeRecord** episodes;

        [NativeTypeName("size_t")]
        public UIntPtr episodes_len;
    }

    internal static unsafe partial class Native
    {
        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("uint32_t")]
        public static extern uint rlmesh_abi_version();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_last_error_message();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern int rlmesh_last_error_is_recoverable();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_last_error_status();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshErrorClass rlmesh_last_error_class();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_last_error_code([NativeTypeName("int32_t *")] int* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_last_error_code_str();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_error_code_str(RLMeshErrorCode code);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_model_error_code_str(RLMeshModelErrorCode code);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_error_class_name(RLMeshErrorClass cls);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_error_class_is_subclass(RLMeshErrorClass cls, RLMeshErrorClass @base);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_build_source_str(RLMeshBuildSource source);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_build_info(RLMeshBuildInfo* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_build_info_repr([NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_current_workflow_edition();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_warning_category_name(RLMeshWarningCategory category);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_warning_category_is_subclass(RLMeshWarningCategory cls, RLMeshWarningCategory @base);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_set_warning_handler([NativeTypeName("RLMeshWarningHandlerFn")] IntPtr fn, void* user_data, [NativeTypeName("void (*)(void *)")] IntPtr free_user_data);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_dtype_str(RLMeshDType dtype);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_ext_dtype_size(RLMeshDType dtype);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_device_str(RLMeshDevice device);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_new([NativeTypeName("const void *")] void* buffer, [NativeTypeName("size_t")] UIntPtr buffer_len, [NativeTypeName("const size_t *")] UIntPtr* shape, [NativeTypeName("size_t")] UIntPtr shape_len, RLMeshDType dtype, RLMeshTensor** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshTensor* rlmesh_tensor_clone([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_tensor_free(RLMeshTensor* tensor);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshDType rlmesh_tensor_dtype([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshDevice rlmesh_tensor_device([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_shape([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("size_t *")] UIntPtr* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_strides([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("size_t *")] UIntPtr* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_ndim([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_size([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_nbytes([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_buffer([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("const void **")] void** out_data, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_tensor_is_contiguous([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_reshape([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("const int64_t *")] long* shape, [NativeTypeName("size_t")] UIntPtr shape_len, RLMeshTensor** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_copy([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, RLMeshTensor** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_tobytes([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_repr([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_from_dlpack([NativeTypeName("struct DLManagedTensorVersioned *")] DLManagedTensorVersioned* obj, RLMeshTensor** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_to_dlpack([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("const RLMeshToDlpackOptions *")] RLMeshToDlpackOptions* options, [NativeTypeName("struct DLManagedTensorVersioned **")] DLManagedTensorVersioned** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tensor_dlpack_device([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, [NativeTypeName("int32_t *")] int* out_type, [NativeTypeName("int32_t *")] int* out_id);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshValueKind rlmesh_value_kind([NativeTypeName("const RLMeshValue *")] RLMeshValue* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshValue* rlmesh_value_clone([NativeTypeName("const RLMeshValue *")] RLMeshValue* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_value_equals([NativeTypeName("const RLMeshValue *")] RLMeshValue* a, [NativeTypeName("const RLMeshValue *")] RLMeshValue* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_value_free(RLMeshValue* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_none(RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_bool([NativeTypeName("_Bool")] byte v, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_int([NativeTypeName("int64_t")] long v, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_float(double v, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_text(RLMeshStr v, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_bytes([NativeTypeName("const uint8_t *")] byte* data, [NativeTypeName("size_t")] UIntPtr len, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_tensor([NativeTypeName("const RLMeshTensor *")] RLMeshTensor* tensor, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_list([NativeTypeName("const RLMeshValue *const *")] RLMeshValue** items, [NativeTypeName("size_t")] UIntPtr n, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_tuple([NativeTypeName("const RLMeshValue *const *")] RLMeshValue** items, [NativeTypeName("size_t")] UIntPtr n, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_dict([NativeTypeName("const RLMeshStr *")] RLMeshStr* keys, [NativeTypeName("const RLMeshValue *const *")] RLMeshValue** values, [NativeTypeName("size_t")] UIntPtr n, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_bool([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_int([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_float([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, double* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_text([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshStr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_bytes([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("const uint8_t **")] byte** out_data, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_as_tensor([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshTensor** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_len([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_get([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("size_t")] UIntPtr i, [NativeTypeName("const RLMeshValue **")] RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_dict_key_at([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("size_t")] UIntPtr i, RLMeshStr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_dict_value_at([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, [NativeTypeName("size_t")] UIntPtr i, [NativeTypeName("const RLMeshValue **")] RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_dict_get([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshStr key, [NativeTypeName("const RLMeshValue **")] RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_value_dict_contains_key([NativeTypeName("const RLMeshValue *")] RLMeshValue* value, RLMeshStr key, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_env_contract_observation_space([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_env_contract_action_space([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("uint64_t")]
        public static extern ulong rlmesh_env_contract_num_envs([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_contract_id([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract, [NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_contract_render_mode([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract, [NativeTypeName("_Bool *")] bool* out_has, [NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshValue *")]
        public static extern RLMeshValue* rlmesh_env_contract_metadata([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvContract* rlmesh_env_contract_clone([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvContract* rlmesh_env_contract_env([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvContract* rlmesh_env_contract_spec([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_env_contract_free(RLMeshEnvContract* contract);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_contract_to_dict([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_contract_repr([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract, [NativeTypeName("struct RLMeshBytes *")] RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_trial_index([NativeTypeName("const RLMeshValue *")] RLMeshValue* options, [NativeTypeName("const size_t *")] UIntPtr* lane, [NativeTypeName("_Bool *")] bool* out_has, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_space_spec_free(RLMeshSpaceSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_bytes_free(RLMeshBytes bytes);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_space_kind_str(RLMeshSpaceKind kind);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_bounds_kind_str(RLMeshBoundsKind kind);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshSpaceSpec* rlmesh_space_spec_clone([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshSpaceKind rlmesh_space_spec_kind([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_spec_shape([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, [NativeTypeName("int64_t *")] long* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshDType rlmesh_space_spec_dtype([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_spec_to_space([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_spec_repr([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_spaces_space_from_spec([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_new([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshSpace* rlmesh_space_clone([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_space_free(RLMeshSpace* space);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_space_spec([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshSpaceKind rlmesh_space_kind([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshDType rlmesh_space_dtype([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_shape([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("int64_t *")] long* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_seed([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("const uint64_t *")] ulong* seed, [NativeTypeName("uint64_t *")] ulong* out_seed);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_sample([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_space_contains([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("const RLMeshValue *")] RLMeshValue* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_space_equals([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* a, [NativeTypeName("const RLMeshSpace *")] RLMeshSpace* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_space_repr([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_box_new(RLMeshBound low, RLMeshBound high, [NativeTypeName("const int64_t *")] long* shape, [NativeTypeName("size_t")] UIntPtr shape_len, [NativeTypeName("const RLMeshBoxOptions *")] RLMeshBoxOptions* options, [NativeTypeName("RLMeshBox **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_box_from_spec([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, [NativeTypeName("RLMeshBox **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_box_low([NativeTypeName("const RLMeshBox *")] RLMeshSpace* box, RLMeshBound* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_box_high([NativeTypeName("const RLMeshBox *")] RLMeshSpace* box, RLMeshBound* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_box_bounds_kind([NativeTypeName("const RLMeshBox *")] RLMeshSpace* box, [NativeTypeName("_Bool *")] bool* out_has, RLMeshBoundsKind* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_box_elementwise_new([NativeTypeName("const RLMeshBound *")] RLMeshBound* low, [NativeTypeName("const RLMeshBound *")] RLMeshBound* high, [NativeTypeName("size_t")] UIntPtr bounds_len, [NativeTypeName("const int64_t *")] long* shape, [NativeTypeName("size_t")] UIntPtr shape_len, [NativeTypeName("const RLMeshBoxOptions *")] RLMeshBoxOptions* options, [NativeTypeName("RLMeshBox **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_discrete_new([NativeTypeName("int64_t")] long n, [NativeTypeName("const RLMeshDiscreteOptions *")] RLMeshDiscreteOptions* options, [NativeTypeName("RLMeshDiscrete **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_discrete_from_spec([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, [NativeTypeName("RLMeshDiscrete **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_discrete_n([NativeTypeName("const RLMeshDiscrete *")] RLMeshSpace* space, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_discrete_start([NativeTypeName("const RLMeshDiscrete *")] RLMeshSpace* space, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_new([NativeTypeName("const RLMeshStr *")] RLMeshStr* spaces_keys, [NativeTypeName("const RLMeshSpaceSpec *const *")] RLMeshSpaceSpec** spaces, [NativeTypeName("size_t")] UIntPtr spaces_len, [NativeTypeName("RLMeshDict **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_from_spec([NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* spec, [NativeTypeName("RLMeshDict **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_len([NativeTypeName("const RLMeshDict *")] RLMeshSpace* dict, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_key_at([NativeTypeName("const RLMeshDict *")] RLMeshSpace* dict, [NativeTypeName("size_t")] UIntPtr i, RLMeshStr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_value_at([NativeTypeName("const RLMeshDict *")] RLMeshSpace* dict, [NativeTypeName("size_t")] UIntPtr i, [NativeTypeName("const RLMeshSpace **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_get([NativeTypeName("const RLMeshDict *")] RLMeshSpace* dict, RLMeshStr key, [NativeTypeName("const RLMeshSpace **")] RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_dict_contains_key([NativeTypeName("const RLMeshDict *")] RLMeshSpace* dict, RLMeshStr key, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_multi_binary_new([NativeTypeName("const int64_t *")] long* shape, [NativeTypeName("size_t")] UIntPtr shape_len, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_multi_discrete_new([NativeTypeName("const int64_t *")] long* nvec, [NativeTypeName("size_t")] UIntPtr nvec_len, RLMeshDType dtype, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_text_new([NativeTypeName("int64_t")] long max_length, [NativeTypeName("int64_t")] long min_length, [NativeTypeName("const char *")] sbyte* charset, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tuple_new([NativeTypeName("const RLMeshSpaceSpec *const *")] RLMeshSpaceSpec** spaces, [NativeTypeName("size_t")] UIntPtr spaces_len, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_multi_discrete_nvec([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("int64_t *")] long* @out, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_text_limits([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("int64_t *")] long* out_min_length, [NativeTypeName("int64_t *")] long* out_max_length, RLMeshStr* out_charset);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tuple_len([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_tuple_get([NativeTypeName("const RLMeshSpace *")] RLMeshSpace* space, [NativeTypeName("size_t")] UIntPtr index, RLMeshSpace** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_image_layout_str(RLMeshImageLayout value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_fit_mode_str(RLMeshFitMode value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_frame_str(RLMeshFrame value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_reference_str(RLMeshReference value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_provenance_str(RLMeshProvenance value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_resample_str(RLMeshResample value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_rotation_encoding_str(RLMeshRotationEncoding value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_stack_pad_str(RLMeshStackPad value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_crop_mode_str(RLMeshCropMode value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_channel_order_str(RLMeshChannelOrder value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_state_container_str(RLMeshStateContainer value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_text_container_str(RLMeshTextContainer value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_state_source_str(RLMeshStateSource value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_advisory_severity_str(RLMeshAdvisorySeverity value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshConcatPart rlmesh_concat_part_role([NativeTypeName("const char *")] sbyte* role);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshConcatPart rlmesh_concat_part_state([NativeTypeName("const RLMeshState *")] RLMeshState* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshConcatPart rlmesh_concat_part_constant([NativeTypeName("const RLMeshConstant *")] RLMeshConstant* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshModelSpec* rlmesh_model_spec_clone([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_model_spec_free(RLMeshModelSpec* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_model_spec_equals([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* a, [NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_repr([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_to_json([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_to_dict([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* value, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_to_metadata([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* value, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_from_dict([NativeTypeName("const RLMeshValue *")] RLMeshValue* data, RLMeshModelSpec** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_from_metadata([NativeTypeName("const RLMeshValue *")] RLMeshValue* metadata, RLMeshModelSpec** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_from_json([NativeTypeName("const char *")] sbyte* payload, RLMeshModelSpec** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTags* rlmesh_env_tags_clone([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_env_tags_free(RLMeshEnvTags* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_env_tags_equals([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* a, [NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_repr([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_to_json([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_to_dict([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* value, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_to_metadata([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* value, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_from_dict([NativeTypeName("const RLMeshValue *")] RLMeshValue* data, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_from_metadata([NativeTypeName("const RLMeshValue *")] RLMeshValue* metadata, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_from_json([NativeTypeName("const char *")] sbyte* payload, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_obs_transform_new([NativeTypeName("RLMeshObsTransformFn")] IntPtr fn, void* user_data, [NativeTypeName("void (*)(void *)")] IntPtr free_user_data, RLMeshObsTransform** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsTransform* rlmesh_obs_transform_clone([NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_obs_transform_free(RLMeshObsTransform* transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_obs_transform_equals([NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* a, [NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("uint64_t")]
        public static extern ulong rlmesh_obs_transform_hash([NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* obs_transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_rotation_transform_new([NativeTypeName("RLMeshRotationTransformFn")] IntPtr fn, void* user_data, [NativeTypeName("void (*)(void *)")] IntPtr free_user_data, RLMeshRotationTransform** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshRotationTransform* rlmesh_rotation_transform_clone([NativeTypeName("const RLMeshRotationTransform *")] RLMeshRotationTransform* transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_rotation_transform_free(RLMeshRotationTransform* transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_rotation_transform_equals([NativeTypeName("const RLMeshRotationTransform *")] RLMeshRotationTransform* a, [NativeTypeName("const RLMeshRotationTransform *")] RLMeshRotationTransform* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("uint64_t")]
        public static extern ulong rlmesh_rotation_transform_hash([NativeTypeName("const RLMeshRotationTransform *")] RLMeshRotationTransform* rotation_transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_new([NativeTypeName("const RLMeshInputNode *")] RLMeshInputNode* input, [NativeTypeName("const RLMeshAction *")] RLMeshAction* output, RLMeshModelSpec** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshInputNode *")]
        public static extern RLMeshInputNode* rlmesh_model_spec_input([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshAction *")]
        public static extern RLMeshAction* rlmesh_model_spec_output([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* spec);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec_with_allow([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* spec, [NativeTypeName("const char *const *")] sbyte** allow, [NativeTypeName("size_t")] UIntPtr allow_len, RLMeshModelSpec** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *const *")]
        public static extern sbyte** rlmesh_model_spec_allow([NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* spec, [NativeTypeName("size_t *")] UIntPtr* out_len);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_new([NativeTypeName("const RLMeshObsNode *")] RLMeshObsNode* observation, [NativeTypeName("const RLMeshAction *")] RLMeshAction* action, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshObsNode *")]
        public static extern RLMeshObsNode* rlmesh_env_tags_observation([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* tags);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshAction *")]
        public static extern RLMeshAction* rlmesh_env_tags_action([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* tags);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_tags_observation_roles([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* tags, RLMeshObservationRoles* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_observation_roles_equals([NativeTypeName("const RLMeshObservationRoles *")] RLMeshObservationRoles* a, [NativeTypeName("const RLMeshObservationRoles *")] RLMeshObservationRoles* b, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_observation_roles_repr([NativeTypeName("const RLMeshObservationRoles *")] RLMeshObservationRoles* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_advisory_equals([NativeTypeName("const RLMeshAdvisory *")] RLMeshAdvisory* a, [NativeTypeName("const RLMeshAdvisory *")] RLMeshAdvisory* b, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_advisory_repr([NativeTypeName("const RLMeshAdvisory *")] RLMeshAdvisory* value, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_action_dim([NativeTypeName("const RLMeshAction *")] RLMeshAction* action, [NativeTypeName("uint32_t *")] uint* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_custom_encoding_is_entrypoint([NativeTypeName("const RLMeshCustomEncoding *")] RLMeshCustomEncoding* encoding, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_custom_encoding_width([NativeTypeName("const RLMeshCustomEncoding *")] RLMeshCustomEncoding* encoding, [NativeTypeName("uint32_t *")] uint* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_rotation_encoding_dims(RLMeshRotationEncoding encoding, [NativeTypeName("uint32_t *")] uint* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_adapters_parts_len();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_adapters_parts_get([NativeTypeName("size_t")] UIntPtr i);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_adapters_rotation_dims_len();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_rotation_dims_get([NativeTypeName("size_t")] UIntPtr i, RLMeshRotationEncoding* encoding, [NativeTypeName("uint32_t *")] uint* dims);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_image_with_size(RLMeshImage* image, [NativeTypeName("uint32_t")] uint size);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_rotation_from_matrix([NativeTypeName("const double[3][3]")] double** rows, [NativeTypeName("double[6]")] double* out_value, RLMeshRotation* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_previous([NativeTypeName("const char *")] sbyte* role, [NativeTypeName("const RLMeshPreviousOptions *")] RLMeshPreviousOptions* options, RLMeshState* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_image([NativeTypeName("const RLMeshImage *")] RLMeshImage* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_state([NativeTypeName("const RLMeshState *")] RLMeshState* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_concat([NativeTypeName("const RLMeshConcat *")] RLMeshConcat* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_text([NativeTypeName("const RLMeshAdaptersText *")] RLMeshAdaptersText* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_custom([NativeTypeName("const RLMeshCustom *")] RLMeshCustom* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_dict([NativeTypeName("const RLMeshStr *")] RLMeshStr* keys, [NativeTypeName("const RLMeshInputNode *")] RLMeshInputNode* children, [NativeTypeName("size_t")] UIntPtr n);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshInputNode rlmesh_input_node_tuple([NativeTypeName("const RLMeshInputNode *")] RLMeshInputNode* children, [NativeTypeName("size_t")] UIntPtr n);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_image_tag([NativeTypeName("const RLMeshImageTag *")] RLMeshImageTag* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_state_tag([NativeTypeName("const RLMeshStateTag *")] RLMeshStateTag* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_text_tag([NativeTypeName("const RLMeshTextTag *")] RLMeshTextTag* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_split([NativeTypeName("const RLMeshSplit *")] RLMeshSplit* leaf);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_dict([NativeTypeName("const RLMeshStr *")] RLMeshStr* keys, [NativeTypeName("const RLMeshObsNode *")] RLMeshObsNode* children, [NativeTypeName("size_t")] UIntPtr n);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshObsNode rlmesh_obs_node_tuple([NativeTypeName("const RLMeshObsNode *")] RLMeshObsNode* children, [NativeTypeName("size_t")] UIntPtr n);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_new([NativeTypeName("const RLMeshAdapterVtable *")] RLMeshAdapterVtable* vtable, void* user_data, RLMeshAdapter** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshAdapter* rlmesh_adapter_clone([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_adapter_free(RLMeshAdapter* adapter);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_resolve([NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* env_tags, [NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* observation_space, [NativeTypeName("const RLMeshSpaceSpec *")] RLMeshSpaceSpec* action_space, [NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* model_spec, [NativeTypeName("const RLMeshResolveOptions *")] RLMeshResolveOptions* options, RLMeshAdapter** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_resolve_from_contract([NativeTypeName("const RLMeshEnvContract *")] RLMeshEnvContract* contract, [NativeTypeName("const RLMeshModelSpec *")] RLMeshModelSpec* model_spec, [NativeTypeName("const RLMeshResolveOptions *")] RLMeshResolveOptions* options, RLMeshAdapter** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_transform_obs(RLMeshAdapter* adapter, [NativeTypeName("const RLMeshValue *")] RLMeshValue* raw_obs, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_transform_action(RLMeshAdapter* adapter, [NativeTypeName("const RLMeshValue *")] RLMeshValue* raw_action, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_observe(RLMeshAdapter* adapter, [NativeTypeName("const RLMeshValue *")] RLMeshValue* raw_obs);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_reset(RLMeshAdapter* adapter);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_explain([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_advisories_len([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_advisories_get([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t")] UIntPtr i, RLMeshAdvisory* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_history_keys_len([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_history_keys_get([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t")] UIntPtr i, RLMeshStr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_history_windows_len([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t *")] UIntPtr* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_history_windows_get([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("size_t")] UIntPtr i, RLMeshStr* out_placement, [NativeTypeName("uint64_t *")] ulong* out_span, [NativeTypeName("uint64_t *")] ulong* out_frame_bytes);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapter_wrap_predict(RLMeshAdapter* adapter, [NativeTypeName("RLMeshPredictFn")] IntPtr predict_fn, void* user_data, [NativeTypeName("void (*)(void *)")] IntPtr free_user_data, RLMeshWrappedPredict** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_wrapped_predict_call(RLMeshWrappedPredict* predict, [NativeTypeName("const RLMeshValue *")] RLMeshValue* raw_obs, RLMeshValue** out_action);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_wrapped_predict_free(RLMeshWrappedPredict* predict);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_tag(RLMeshEnv* env, [NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* tags, [NativeTypeName("const RLMeshTagOptions *")] RLMeshTagOptions* options);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_adapters_tag_vector(RLMeshVectorEnv* env, [NativeTypeName("const RLMeshEnvTags *")] RLMeshEnvTags* tags, [NativeTypeName("const RLMeshTagOptions *")] RLMeshTagOptions* options);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_unregister_entrypoint([NativeTypeName("const char *")] sbyte* entrypoint, [NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* expected, [NativeTypeName("_Bool *")] bool* out_removed);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_register_entrypoint([NativeTypeName("const char *")] sbyte* entrypoint, [NativeTypeName("const RLMeshObsTransform *")] RLMeshObsTransform* transform);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const char *")]
        public static extern sbyte* rlmesh_transport_str(RLMeshTransport transport);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_new([NativeTypeName("const RLMeshEnvVtable *")] RLMeshEnvVtable* env_vtable, void* user_data, [NativeTypeName("const RLMeshEnvAttributes *")] RLMeshEnvAttributes* env_attributes, RLMeshEnv** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_vector_env_new([NativeTypeName("const RLMeshVectorEnvVtable *")] RLMeshVectorEnvVtable* vector_env_vtable, void* user_data, [NativeTypeName("const RLMeshVectorEnvAttributes *")] RLMeshVectorEnvAttributes* vector_env_attributes, RLMeshVectorEnv** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_close(RLMeshEnv* env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_vector_env_close(RLMeshVectorEnv* vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_env_free(RLMeshEnv* env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_vector_env_free(RLMeshVectorEnv* vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_new([NativeTypeName("RLMeshEnv *const *")] RLMeshEnv** envs, [NativeTypeName("size_t")] UIntPtr envs_len, [NativeTypeName("const RLMeshEnvServerOptions *")] RLMeshEnvServerOptions* options, RLMeshEnvServer** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_new_vector(RLMeshVectorEnv* vector_env, [NativeTypeName("const RLMeshEnvServerOptions *")] RLMeshEnvServerOptions* options, RLMeshEnvServer** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_address([NativeTypeName("const RLMeshEnvServer *")] RLMeshEnvServer* env_server, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshEnvContract *")]
        public static extern RLMeshEnvContract* rlmesh_env_server_env_contract([NativeTypeName("const RLMeshEnvServer *")] RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_start(RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_serve(RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_wait(RLMeshEnvServer* env_server, [NativeTypeName("const double *")] double* timeout, [NativeTypeName("_Bool *")] bool* out_stopped);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_shutdown(RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_server_repr([NativeTypeName("const RLMeshEnvServer *")] RLMeshEnvServer* env_server, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_env_server_free(RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_cancel_token_new(RLMeshExtCancelToken** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_cancel_token_clone([NativeTypeName("const RLMeshExtCancelToken *")] RLMeshExtCancelToken* ext_cancel_token, RLMeshExtCancelToken** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_cancel_token_cancel(RLMeshExtCancelToken* ext_cancel_token);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_ext_cancel_token_free(RLMeshExtCancelToken* ext_cancel_token);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_new([NativeTypeName("const RLMeshEnvFactoryVtable *")] RLMeshEnvFactoryVtable* vtable, void* user_data, [NativeTypeName("const RLMeshEnvFactoryAttributes *")] RLMeshEnvFactoryAttributes* attributes, RLMeshEnvFactory** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_env_factory_free(RLMeshEnvFactory* factory);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_prepare(RLMeshEnvFactory* factory);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_make(RLMeshEnvFactory* factory, [NativeTypeName("const RLMeshValue *")] RLMeshValue* kwargs, RLMeshEnv** out_env, RLMeshVectorEnv** out_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_tags_for(RLMeshEnvFactory* factory, [NativeTypeName("const RLMeshValue *")] RLMeshValue* @params, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_enumerate_variants(RLMeshEnvFactory* factory, RLMeshVariantList** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_close(RLMeshEnvFactory* factory);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_serve(RLMeshEnvFactory* factory, [NativeTypeName("const char *")] sbyte* address, [NativeTypeName("const RLMeshEnvFactoryServeOptions *")] RLMeshEnvFactoryServeOptions* options);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_env_factory_start(RLMeshEnvFactory* factory, [NativeTypeName("const char *")] sbyte* address, [NativeTypeName("const RLMeshEnvFactoryServeOptions *")] RLMeshEnvFactoryServeOptions* options, RLMeshEnvServer** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_tags([NativeTypeName("const RLMeshEnvFactory *")] RLMeshEnvFactory* env_factory, RLMeshEnvTags** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_reset_options([NativeTypeName("const RLMeshEnvFactory *")] RLMeshEnvFactory* env_factory, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_tag_params([NativeTypeName("const RLMeshEnvFactory *")] RLMeshEnvFactory* env_factory, RLMeshTagParamList** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_workflow_edition([NativeTypeName("const RLMeshEnvFactory *")] RLMeshEnvFactory* env_factory, [NativeTypeName("_Bool *")] bool* out_has, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_variant_list_len([NativeTypeName("const RLMeshVariantList *")] RLMeshVariantList* variant_list);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshVariant *")]
        public static extern RLMeshVariant* rlmesh_variant_list_get([NativeTypeName("const RLMeshVariantList *")] RLMeshVariantList* variant_list, [NativeTypeName("size_t")] UIntPtr index);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_variant_list_free(RLMeshVariantList* variant_list);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_tag_param_list_len([NativeTypeName("const RLMeshTagParamList *")] RLMeshTagParamList* tag_param_list);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshTagParam *")]
        public static extern RLMeshTagParam* rlmesh_tag_param_list_get([NativeTypeName("const RLMeshTagParamList *")] RLMeshTagParamList* tag_param_list, [NativeTypeName("size_t")] UIntPtr index);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_tag_param_list_free(RLMeshTagParamList* tag_param_list);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_new([NativeTypeName("const char *")] sbyte* id, [NativeTypeName("const RLMeshValue *")] RLMeshValue* @params, [NativeTypeName("const RLMeshVariantOptions *")] RLMeshVariantOptions* options, RLMeshVariant** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_variant_free(RLMeshVariant* variant);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_id([NativeTypeName("const RLMeshVariant *")] RLMeshVariant* variant, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshValue *")]
        public static extern RLMeshValue* rlmesh_variant_params([NativeTypeName("const RLMeshVariant *")] RLMeshVariant* variant);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshValue *")]
        public static extern RLMeshValue* rlmesh_variant_metadata([NativeTypeName("const RLMeshVariant *")] RLMeshVariant* variant);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_repr([NativeTypeName("const RLMeshVariant *")] RLMeshVariant* variant, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_variant_equals([NativeTypeName("const RLMeshVariant *")] RLMeshVariant* a, [NativeTypeName("const RLMeshVariant *")] RLMeshVariant* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_set_id(RLMeshVariant* variant, [NativeTypeName("const char *")] sbyte* id);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_set_params(RLMeshVariant* variant, [NativeTypeName("const RLMeshValue *")] RLMeshValue* @params);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_set_metadata(RLMeshVariant* variant, [NativeTypeName("const RLMeshValue *")] RLMeshValue* metadata);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_variant_sink_push(RLMeshVariantSink* sink, RLMeshVariant* variant);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_state_get([NativeTypeName("const RLMeshEpisodeState *")] RLMeshEpisodeState* episode_state, [NativeTypeName("const char *")] sbyte* key, void** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_state_set(RLMeshEpisodeState* episode_state, [NativeTypeName("const char *")] sbyte* key, void* value, [NativeTypeName("void (*)(void *)")] IntPtr free_value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_state_remove(RLMeshEpisodeState* episode_state, [NativeTypeName("const char *")] sbyte* key);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_episode_state_len([NativeTypeName("const RLMeshEpisodeState *")] RLMeshEpisodeState* episode_state);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_serve(RLMeshModel* model, [NativeTypeName("const char *")] sbyte* address, [NativeTypeName("const RLMeshModelServeOptions *")] RLMeshModelServeOptions* options);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_new([NativeTypeName("const RLMeshModelVtable *")] RLMeshModelVtable* vtable, void* user_data, [NativeTypeName("const RLMeshModelOptions *")] RLMeshModelOptions* options, RLMeshModel** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_from_config([NativeTypeName("const RLMeshModelVtable *")] RLMeshModelVtable* model_vtable, void* user_data, [NativeTypeName("const RLMeshValue *")] RLMeshValue* config, RLMeshModel** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_from_address([NativeTypeName("const char *")] sbyte* address, RLMeshModel** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_clone([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, RLMeshModel** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_close(RLMeshModel* model);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_model_free(RLMeshModel* model);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_spec([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, RLMeshSpecArg* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_model_allow_fusion([NativeTypeName("const RLMeshModel *")] RLMeshModel* model);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_model_native_chunk([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, [NativeTypeName("_Bool *")] bool* out_has, [NativeTypeName("uint32_t *")] uint* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_workflow_edition([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, [NativeTypeName("_Bool *")] bool* out_has, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_predict_seed([NativeTypeName("int64_t")] long episode_seed, [NativeTypeName("uint64_t")] ulong predict_index);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_vtable_describe([NativeTypeName("const RLMeshModelVtable *")] RLMeshModelVtable* vtable, [NativeTypeName("const RLMeshDescribeOptions *")] RLMeshDescribeOptions* options, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_vtable_describe_json([NativeTypeName("const RLMeshModelVtable *")] RLMeshModelVtable* vtable, [NativeTypeName("const RLMeshDescribeJsonOptions *")] RLMeshDescribeJsonOptions* options, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_describe([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, [NativeTypeName("const RLMeshDescribeOptions *")] RLMeshDescribeOptions* options, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_describe_json([NativeTypeName("const RLMeshModel *")] RLMeshModel* model, [NativeTypeName("const RLMeshDescribeJsonOptions *")] RLMeshDescribeJsonOptions* options, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_describe(RLMeshEnvFactory* env_factory, [NativeTypeName("const RLMeshDescribeOptions *")] RLMeshDescribeOptions* options, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_env_factory_describe_json(RLMeshEnvFactory* env_factory, [NativeTypeName("const RLMeshDescribeJsonOptions *")] RLMeshDescribeJsonOptions* options, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_env_server_describe_json([NativeTypeName("const RLMeshEnvServer *")] RLMeshEnvServer* env_server, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshModelTarget rlmesh_model_target(RLMeshModel* model);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_address_target([NativeTypeName("const char *")] sbyte* value);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_env_server_target([NativeTypeName("const RLMeshEnvServer *")] RLMeshEnvServer* env_server);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_env_target([NativeTypeName("const RLMeshEnv *")] RLMeshEnv* env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_vector_env_target([NativeTypeName("const RLMeshVectorEnv *")] RLMeshVectorEnv* vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_env_factory_target([NativeTypeName("const RLMeshEnvFactory *")] RLMeshEnvFactory* env_factory);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_result_equals([NativeTypeName("const RLMeshEpisodeResult *")] RLMeshEpisodeResult* a, [NativeTypeName("const RLMeshEpisodeResult *")] RLMeshEpisodeResult* b, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_result_repr([NativeTypeName("const RLMeshEpisodeResult *")] RLMeshEpisodeResult* episode_result, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_new([NativeTypeName("const RLMeshRunResultOptions *")] RLMeshRunResultOptions* options, RLMeshRunResult** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshRunResult* rlmesh_run_result_clone([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_run_result_free(RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_run_result_episodes_len([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_run_result_num_episodes([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_run_result_total_steps([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_run_result_mean_reward([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_run_result_success_rate([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, double* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_episodes_get([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, [NativeTypeName("size_t")] UIntPtr i, RLMeshEpisodeResult* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_run_result_telemetry_len([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_telemetry_get([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, [NativeTypeName("size_t")] UIntPtr i, RLMeshTelemetryRow** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_format_telemetry([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_repr([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_run_result_equals([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* a, [NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_save([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, [NativeTypeName("const char *")] sbyte* path, [NativeTypeName("const RLMeshRunResultSaveOptions *")] RLMeshRunResultSaveOptions* options, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_load([NativeTypeName("const char *")] sbyte* path, RLMeshRunResult** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_new([NativeTypeName("const char *")] sbyte* op, [NativeTypeName("const char *")] sbyte* component, [NativeTypeName("const char *")] sbyte* metric, [NativeTypeName("const char *")] sbyte* unit, [NativeTypeName("int64_t")] long count, double avg, double p50, double p95, double p99, RLMeshTelemetryRow** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_op([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_component([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_metric([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_unit([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_telemetry_row_count([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_telemetry_row_avg([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_telemetry_row_p50([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_telemetry_row_p95([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_telemetry_row_p99([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_telemetry_row_repr([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* telemetry_row, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_telemetry_row_equals([NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* a, [NativeTypeName("const RLMeshTelemetryRow *")] RLMeshTelemetryRow* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_telemetry_row_free(RLMeshTelemetryRow* telemetry_row);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_context_new([NativeTypeName("const RLMeshRunContextVtable *")] RLMeshRunContextVtable* run_context_vtable, void* user_data, RLMeshRunContext** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshRunContext* rlmesh_run_context_clone([NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* run_context);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_run_context_free(RLMeshRunContext* run_context);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_context_contract([NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* run_context, RLMeshEnvContract** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_context_num_envs([NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* run_context, [NativeTypeName("uint64_t *")] ulong* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_step_event_equals([NativeTypeName("const RLMeshStepEvent *")] RLMeshStepEvent* a, [NativeTypeName("const RLMeshStepEvent *")] RLMeshStepEvent* b, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_step_event_repr([NativeTypeName("const RLMeshStepEvent *")] RLMeshStepEvent* step_event, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_new([NativeTypeName("const RLMeshRunHooksVtable *")] RLMeshRunHooksVtable* run_hooks_vtable, void* user_data, RLMeshRunHooks** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_run_hooks_free(RLMeshRunHooks* run_hooks);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_on_run_start(RLMeshRunHooks* hooks, [NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* context);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_on_episode_start(RLMeshRunHooks* hooks, [NativeTypeName("int64_t")] long episode, [NativeTypeName("const int64_t *")] long* seed);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_on_step(RLMeshRunHooks* hooks, [NativeTypeName("const RLMeshStepEvent *")] RLMeshStepEvent* @event);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_on_episode_end(RLMeshRunHooks* hooks, [NativeTypeName("const RLMeshEpisodeResult *")] RLMeshEpisodeResult* result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_hooks_on_run_end(RLMeshRunHooks* hooks, [NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_view_validate([NativeTypeName("const RLMeshView *")] RLMeshView* view);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_view_parse([NativeTypeName("const char *")] sbyte* text, RLMeshView* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_view_equals([NativeTypeName("const RLMeshView *")] RLMeshView* a, [NativeTypeName("const RLMeshView *")] RLMeshView* b, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_view_repr([NativeTypeName("const RLMeshView *")] RLMeshView* view, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run(RLMeshModelTarget model, RLMeshEnvTarget env, [NativeTypeName("const RLMeshRunOptions *")] RLMeshRunOptions* options, RLMeshRunResult** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_run(RLMeshModel* model, RLMeshEnvTarget env_or_address, [NativeTypeName("const RLMeshRunOptions *")] RLMeshRunOptions* options, RLMeshRunResult** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_bind(RLMeshModelTarget model, RLMeshEnvTarget target, [NativeTypeName("const RLMeshBindOptions *")] RLMeshBindOptions* options, RLMeshBinding** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_model_bind(RLMeshModel* model, RLMeshEnvTarget target, [NativeTypeName("const RLMeshBindOptions *")] RLMeshBindOptions* options, RLMeshBinding** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_reset(RLMeshBinding* binding, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("const RLMeshValue *")] RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_step(RLMeshBinding* binding, [NativeTypeName("const RLMeshValue *")] RLMeshValue* action, RLMeshValue** out_observation, double* out_reward, [NativeTypeName("_Bool *")] bool* out_terminated, [NativeTypeName("_Bool *")] bool* out_truncated, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_done([NativeTypeName("const RLMeshBinding *")] RLMeshBinding* binding, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_contract(RLMeshBinding* binding, RLMeshEnvContract** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_num_envs(RLMeshBinding* binding, [NativeTypeName("uint64_t *")] ulong* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_render_mode(RLMeshBinding* binding, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_render(RLMeshBinding* binding, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_close(RLMeshBinding* binding);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_binding_free(RLMeshBinding* binding);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_remote_env_reset(RLMeshRemoteEnv* remote_env, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("const RLMeshValue *")] RLMeshValue* options, [NativeTypeName("const RLMeshExtCancelToken *")] RLMeshExtCancelToken* cancel, RLMeshValue** out_observation, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_remote_env_step(RLMeshRemoteEnv* remote_env, [NativeTypeName("const RLMeshValue *")] RLMeshValue* action, [NativeTypeName("const RLMeshExtCancelToken *")] RLMeshExtCancelToken* cancel, RLMeshValue** out_observation, double* out_reward, [NativeTypeName("_Bool *")] bool* out_terminated, [NativeTypeName("_Bool *")] bool* out_truncated, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_remote_vector_env_reset(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("size_t")] UIntPtr seed_len, [NativeTypeName("const RLMeshValue *")] RLMeshValue* options, [NativeTypeName("const RLMeshExtCancelToken *")] RLMeshExtCancelToken* cancel, RLMeshValue** out_observation, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_remote_vector_env_step(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("const RLMeshValue *")] RLMeshValue* actions, [NativeTypeName("const RLMeshExtCancelToken *")] RLMeshExtCancelToken* cancel, RLMeshValue** out_observation, double* out_rewards, [NativeTypeName("_Bool *")] bool* out_terminated, [NativeTypeName("_Bool *")] bool* out_truncated, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_new([NativeTypeName("const RLMeshRemoteEnvOptions *")] RLMeshRemoteEnvOptions* options, RLMeshRemoteEnv** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_new([NativeTypeName("const RLMeshRemoteEnvOptions *")] RLMeshRemoteEnvOptions* options, RLMeshRemoteVectorEnv** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_remote_env_target(RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshEnvTarget rlmesh_remote_vector_env_target(RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_address([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_env_id([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_selected_workflow_edition([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshEnvContract *")]
        public static extern RLMeshEnvContract* rlmesh_remote_env_env_contract([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshValue *")]
        public static extern RLMeshValue* rlmesh_remote_env_metadata([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_env_observation_space([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_env_action_space([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_remote_env_observation_space_spec([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_remote_env_action_space_spec([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_render_mode([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env, [NativeTypeName("_Bool *")] bool* out_has, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_cancel_handle([NativeTypeName("const RLMeshRemoteEnv *")] RLMeshRemoteEnv* remote_env, RLMeshExtCancelToken** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_render(RLMeshRemoteEnv* remote_env, [NativeTypeName("uint64_t")] ulong env_index, RLMeshValue** out_frame);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_shutdown(RLMeshRemoteEnv* remote_env, [NativeTypeName("const char *")] sbyte* reason, [NativeTypeName("_Bool *")] bool* out_accepted);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_close(RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_remote_env_free(RLMeshRemoteEnv* remote_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_address([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_env_id([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_selected_workflow_edition([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshEnvContract *")]
        public static extern RLMeshEnvContract* rlmesh_remote_vector_env_env_contract([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshValue *")]
        public static extern RLMeshValue* rlmesh_remote_vector_env_metadata([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_vector_env_observation_space([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_vector_env_action_space([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_remote_vector_env_observation_space_spec([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpaceSpec *")]
        public static extern RLMeshSpaceSpec* rlmesh_remote_vector_env_action_space_spec([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_render_mode([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("_Bool *")] bool* out_has, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_cancel_handle([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env, RLMeshExtCancelToken** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_render(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("uint64_t")] ulong env_index, RLMeshValue** out_frame);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_shutdown(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("const char *")] sbyte* reason, [NativeTypeName("_Bool *")] bool* out_accepted);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_close(RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_remote_vector_env_free(RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("uint64_t")]
        public static extern ulong rlmesh_remote_vector_env_num_envs([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_vector_env_single_observation_space([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("const RLMeshSpace *")]
        public static extern RLMeshSpace* rlmesh_remote_vector_env_single_action_space([NativeTypeName("const RLMeshRemoteVectorEnv *")] RLMeshRemoteVectorEnv* remote_vector_env);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_reset(RLMeshRemoteEnv* remote_env, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("const RLMeshValue *")] RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_env_step(RLMeshRemoteEnv* remote_env, [NativeTypeName("const RLMeshValue *")] RLMeshValue* action, RLMeshValue** out_observation, double* out_reward, [NativeTypeName("_Bool *")] bool* out_terminated, [NativeTypeName("_Bool *")] bool* out_truncated, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_reset(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("size_t")] UIntPtr seed_len, [NativeTypeName("const RLMeshValue *")] RLMeshValue* options, RLMeshValue** out_observation, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_remote_vector_env_step(RLMeshRemoteVectorEnv* remote_vector_env, [NativeTypeName("const RLMeshValue *")] RLMeshValue* actions, RLMeshValue** out_observation, double* out_rewards, [NativeTypeName("_Bool *")] bool* out_terminated, [NativeTypeName("_Bool *")] bool* out_truncated, [NativeTypeName("size_t")] UIntPtr capacity, [NativeTypeName("size_t *")] UIntPtr* out_len, RLMeshValue** out_info);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_read_item_role([NativeTypeName("const char *")] sbyte* role, RLMeshReadItem** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_read_item_image([NativeTypeName("const RLMeshImage *")] RLMeshImage* image, RLMeshReadItem** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_read_item_state([NativeTypeName("const RLMeshState *")] RLMeshState* state, RLMeshReadItem** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_read_item_text([NativeTypeName("const RLMeshAdaptersText *")] RLMeshAdaptersText* adapters_text, RLMeshReadItem** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_read_item_free(RLMeshReadItem* read_item);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_reader_new([NativeTypeName("const RLMeshAdapter *")] RLMeshAdapter* adapter, [NativeTypeName("const char *const *")] sbyte** roles, [NativeTypeName("size_t")] UIntPtr roles_len, RLMeshReader** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_reader_free(RLMeshReader* reader);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_reader_roles_len([NativeTypeName("const RLMeshReader *")] RLMeshReader* reader);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_reader_roles_get([NativeTypeName("const RLMeshReader *")] RLMeshReader* reader, [NativeTypeName("size_t")] UIntPtr i, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_reader_call([NativeTypeName("const RLMeshReader *")] RLMeshReader* reader, [NativeTypeName("const RLMeshValue *")] RLMeshValue* observation, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_read(RLMeshBinding* binding, [NativeTypeName("const RLMeshValue *")] RLMeshValue* observation, [NativeTypeName("const RLMeshReadItem *")] RLMeshReadItem* item, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_context_read([NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* context, [NativeTypeName("const RLMeshValue *")] RLMeshValue* observation, [NativeTypeName("const RLMeshReadItem *")] RLMeshReadItem* item, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_step_event_read([NativeTypeName("const RLMeshStepEvent *")] RLMeshStepEvent* step_event, [NativeTypeName("const RLMeshReadItem *")] RLMeshReadItem* read_item, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_new([NativeTypeName("const char *const *")] sbyte** roles, [NativeTypeName("size_t")] UIntPtr roles_len, [NativeTypeName("const char *")] sbyte* render_label, [NativeTypeName("RLMeshRenderFn")] IntPtr render, void* user_data, [NativeTypeName("void (*)(void *)")] IntPtr free_user_data, RLMeshFrameSources** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_binding_frame_sources(RLMeshBinding* binding, RLMeshFrameSources** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_context_frame_sources([NativeTypeName("const RLMeshRunContext *")] RLMeshRunContext* run_context, RLMeshFrameSources** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_frame_sources_roles_len([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_roles_get([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, [NativeTypeName("size_t")] UIntPtr i, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_render_label([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, RLMeshBytes* @out, [NativeTypeName("_Bool *")] bool* out_has);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_render([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, RLMeshValue** @out, [NativeTypeName("_Bool *")] bool* out_has);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_frame_sources_labels_len([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_labels_get([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, [NativeTypeName("size_t")] UIntPtr i, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_default_label([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, RLMeshBytes* @out, [NativeTypeName("_Bool *")] bool* out_has);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_frame_sources_equals([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* a, [NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_frame_sources_repr([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_frame_sources_free(RLMeshFrameSources* frame_sources);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_ext_frame_sources_has_render([NativeTypeName("const RLMeshFrameSources *")] RLMeshFrameSources* frame_sources);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_run_result_advisories_len([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_run_result_advisories_get([NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* run_result, [NativeTypeName("size_t")] UIntPtr i, RLMeshAdvisory* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_callback_set_error([NativeTypeName("const char *")] sbyte* message, [NativeTypeName("_Bool")] byte recoverable);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_callback_set_error_class(RLMeshErrorClass error_class, [NativeTypeName("const char *")] sbyte* message);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_ext_callback_set_error_payload(void* payload, [NativeTypeName("void (*)(void *)")] IntPtr free_payload);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void* rlmesh_ext_last_error_take_payload();

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_ext_callback_set_tagged_error_payload(void* payload, [NativeTypeName("void (*)(void *)")] IntPtr free_payload, [NativeTypeName("const void *")] void* tag);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_ext_last_error_take_payload_owned(void** out_payload, [NativeTypeName("void (**)(void *)")] IntPtr* out_free_payload, [NativeTypeName("const void **")] void** out_tag);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_new([NativeTypeName("const RLMeshRecorderOptions *")] RLMeshRecorderOptions* options, RLMeshRecorder** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_add(RLMeshRecorder* recorder, [NativeTypeName("const RLMeshRunResult *")] RLMeshRunResult* result, [NativeTypeName("const char *")] sbyte* model, [NativeTypeName("const char *")] sbyte* env, [NativeTypeName("const RLMeshRecorderAddOptions *")] RLMeshRecorderAddOptions* options, RLMeshWorkloadRecord** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_capture(RLMeshRecorder* recorder, [NativeTypeName("const char *")] sbyte* model, [NativeTypeName("const char *")] sbyte* env, [NativeTypeName("const RLMeshRecorderCaptureOptions *")] RLMeshRecorderCaptureOptions* options, RLMeshRunHooks** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_to_dict([NativeTypeName("const RLMeshRecorder *")] RLMeshRecorder* recorder, [NativeTypeName("const RLMeshRecorderToDictOptions *")] RLMeshRecorderToDictOptions* options, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_export([NativeTypeName("const RLMeshRecorder *")] RLMeshRecorder* recorder, [NativeTypeName("const char *")] sbyte* path, [NativeTypeName("const RLMeshRecorderExportOptions *")] RLMeshRecorderExportOptions* options, RLMeshBytes* out_path);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_result_set_id([NativeTypeName("const RLMeshRecorder *")] RLMeshRecorder* recorder, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_recorder_workloads_len([NativeTypeName("const RLMeshRecorder *")] RLMeshRecorder* recorder);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_workloads_get([NativeTypeName("const RLMeshRecorder *")] RLMeshRecorder* recorder, [NativeTypeName("size_t")] UIntPtr i, RLMeshWorkloadRecord** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_recorder_close(RLMeshRecorder* recorder);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_recorder_free(RLMeshRecorder* recorder);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_new([NativeTypeName("const char *")] sbyte* camera, [NativeTypeName("const char *")] sbyte* kind, [NativeTypeName("const char *")] sbyte* path, [NativeTypeName("const char *")] sbyte* format, [NativeTypeName("const RLMeshMediaRefOptions *")] RLMeshMediaRefOptions* options, RLMeshMediaRef** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_camera([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_kind([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_path([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_format([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_media_ref_frame_count([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_media_ref_width([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_media_ref_height([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_media_ref_fps([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, double* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_to_dict([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_media_ref_equals([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* a, [NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_media_ref_repr([NativeTypeName("const RLMeshMediaRef *")] RLMeshMediaRef* media_ref, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_media_ref_free(RLMeshMediaRef* media_ref);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_record_new([NativeTypeName("int64_t")] long index, [NativeTypeName("const int64_t *")] long* seed, [NativeTypeName("int64_t")] long steps, double reward, [NativeTypeName("_Bool")] byte terminated, [NativeTypeName("_Bool")] byte truncated, [NativeTypeName("const RLMeshEpisodeRecordOptions *")] RLMeshEpisodeRecordOptions* options, RLMeshEpisodeRecord** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_episode_record_index([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_seed([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record, [NativeTypeName("int64_t *")] long* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_episode_record_steps([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_episode_record_reward([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_terminated([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_truncated([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_success([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record, [NativeTypeName("_Bool *")] bool* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_episode_record_duration_s([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_included_in_metrics([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_episode_record_media_len([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_record_media_get([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record, [NativeTypeName("size_t")] UIntPtr i, RLMeshMediaRef** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_record_to_dict([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_episode_record_equals([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* a, [NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_episode_record_repr([NativeTypeName("const RLMeshEpisodeRecord *")] RLMeshEpisodeRecord* episode_record, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_episode_record_free(RLMeshEpisodeRecord* episode_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_new([NativeTypeName("const char *")] sbyte* model, [NativeTypeName("const char *")] sbyte* env, [NativeTypeName("const char *")] sbyte* task, [NativeTypeName("const RLMeshWorkloadRecordOptions *")] RLMeshWorkloadRecordOptions* options, RLMeshWorkloadRecord** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_model([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_env([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_task([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_config([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("size_t")]
        public static extern UIntPtr rlmesh_workload_record_episodes_len([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_episodes_get([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, [NativeTypeName("size_t")] UIntPtr i, RLMeshEpisodeRecord** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern double rlmesh_workload_record_mean_reward([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_workload_record_success_rate([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, double* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("int64_t")]
        public static extern long rlmesh_workload_record_total_steps([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_to_dict([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshValue** @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        [return: NativeTypeName("_Bool")]
        public static extern byte rlmesh_workload_record_equals([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* a, [NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* b);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern RLMeshStatus rlmesh_workload_record_repr([NativeTypeName("const RLMeshWorkloadRecord *")] RLMeshWorkloadRecord* workload_record, RLMeshBytes* @out);

        [DllImport("rlmesh_capi", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern void rlmesh_workload_record_free(RLMeshWorkloadRecord* workload_record);
    }

    /// <summary>Defines the type of a member as it was used in the native signature.</summary>
    [AttributeUsage(AttributeTargets.Struct | AttributeTargets.Enum | AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter | AttributeTargets.ReturnValue, AllowMultiple = false, Inherited = true)]
    [Conditional("DEBUG")]
    internal sealed partial class NativeTypeNameAttribute : Attribute
    {
        private readonly string _name;

        /// <summary>Initializes a new instance of the <see cref="NativeTypeNameAttribute" /> class.</summary>
        /// <param name="name">The name of the type that was used in the native signature.</param>
        public NativeTypeNameAttribute(string name)
        {
            _name = name;
        }

        /// <summary>Gets the name of the type that was used in the native signature.</summary>
        public string Name => _name;
    }

    /// <summary>Defines the annotation found in a native declaration.</summary>
    [AttributeUsage(AttributeTargets.Struct | AttributeTargets.Enum | AttributeTargets.Property | AttributeTargets.Field | AttributeTargets.Parameter | AttributeTargets.ReturnValue, AllowMultiple = true, Inherited = false)]
    [Conditional("DEBUG")]
    internal sealed partial class NativeAnnotationAttribute : Attribute
    {
        private readonly string _annotation;

        /// <summary>Initializes a new instance of the <see cref="NativeAnnotationAttribute" /> class.</summary>
        /// <param name="annotation">The annotation that was used in the native declaration.</param>
        public NativeAnnotationAttribute(string annotation)
        {
            _annotation = annotation;
        }

        /// <summary>Gets the annotation that was used in the native declaration.</summary>
        public string Annotation => _annotation;
    }
}
