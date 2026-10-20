using System;
using RLMesh.Spaces;
using UnityEngine;

namespace RLMesh.Unity.Samples
{
    public class RLMeshReachTargetBehaviour : RLMeshEnvBehaviour
    {
        public float TargetRange = 500;
        public float StepSize = 25;
        public int MaxSteps = 100;
        private Vector3 start;
        private System.Random random = new System.Random();
        private float target;
        private int steps;
        protected override void Start() { start = transform.position; base.Start(); }
        protected override Space ObservationSpace => new Box(float.NegativeInfinity, float.PositiveInfinity, new long[] { 1 });
        protected override Space ActionSpace => new Discrete(3);
        protected override ResetResult ResetEnv(long? seed, Metadata options)
        {
            if (TargetRange <= 0 || StepSize <= 0 || MaxSteps < 1) throw new EnvironmentException("Reach Target settings must be positive");
            if (seed.HasValue) random = new System.Random(unchecked((int)seed.Value));
            transform.position = start;
            target = (float)((random.NextDouble() * 2 - 1) * TargetRange);
            steps = 0;
            return new ResetResult(Observe(), new Metadata());
        }
        protected override StepResult StepEnv(Value action)
        {
            if (action == null || action.Kind != ValueKind.Int || action.AsInt() < 0 || action.AsInt() > 2)
                throw new RecoverableEnvironmentException("step needs an action: 0, 1 or 2");
            transform.position += Vector3.right * ((action.AsInt() - 1) * StepSize);
            steps++;
            float distance = Math.Abs(Offset());
            bool reached = distance <= StepSize / 2;
            return new StepResult(Observe(), -distance, reached, !reached && steps >= MaxSteps, new Metadata());
        }
        private float Offset() => target - (transform.position.x - start.x);
        private Value Observe()
        {
            using (var tensor = new Tensor(BitConverter.GetBytes(Offset()), new long[] { 1 }, DType.Float32)) return tensor;
        }
    }
}
