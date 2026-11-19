using System;
using RLMesh.Spaces;
using UnityEngine;
using Space = RLMesh.Spaces.Space;

namespace RLMesh.Unity.Samples
{
    public sealed class Agent : RLMeshEnvBehaviour
    {
        public Transform target;
        public float stepSize = 0.5f;
        public int maxSteps = 100;

        Vector3 start;
        float goalX;
        int steps;

        void Awake()
        {
            if (EnvId == "Unity-v0")
                EnvId = "Unity-ReachTarget-v0";
        }

        protected override void Start()
        {
            if (target == null)
                throw new EnvironmentException("Agent needs a target Transform");
            start = transform.position;
            base.Start();
        }

        protected override Space ObservationSpace =>
            new Box(float.NegativeInfinity, float.PositiveInfinity, new long[] { 1 });

        protected override Space ActionSpace => new Discrete(3);

        protected override ResetResult ResetEnv(long? seed, Metadata options)
        {
            if (stepSize <= 0 || maxSteps < 1)
                throw new EnvironmentException("Agent stepSize and maxSteps must be positive");
            if (target == null)
                throw new EnvironmentException("Agent needs a target Transform");

            transform.position = start;
            goalX = target.position.x;
            steps = 0;
            return new ResetResult(Observe(), MakeInfo());
        }

        protected override StepResult StepEnv(Value action)
        {
            if (action == null || action.Kind != ValueKind.Int || action.AsInt() < 0 || action.AsInt() > 2)
                throw new RecoverableEnvironmentException("step needs an action: 0, 1 or 2");

            transform.position += Vector3.right * ((action.AsInt() - 1) * stepSize);
            steps++;

            float distance = Math.Abs(Offset());
            bool reached = distance <= stepSize / 2f;
            return new StepResult(Observe(), -distance, reached, !reached && steps >= maxSteps, MakeInfo());
        }

        float Offset() => goalX - transform.position.x;

        Value Observe()
        {
            using (var tensor = new Tensor(BitConverter.GetBytes(Offset()), new long[] { 1 }, DType.Float32))
                return tensor;
        }

        Metadata MakeInfo()
        {
            var info = new Metadata();
            info["distance"] = Math.Abs(Offset());
            info["steps"] = (long)steps;
            info["agent_x"] = transform.position.x;
            info["target_x"] = goalX;
            return info;
        }
    }
}
