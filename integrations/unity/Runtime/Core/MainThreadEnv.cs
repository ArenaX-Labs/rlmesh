using System;
using RLMesh.Spaces;

namespace RLMesh.Unity
{
    public sealed class MainThreadEnv : IEnv
    {
        private readonly MainThreadDispatcher dispatcher;
        private readonly IEnv env;
        public MainThreadEnv(MainThreadDispatcher dispatcher, IEnv env)
        {
            this.dispatcher = dispatcher ?? throw new ArgumentNullException(nameof(dispatcher));
            this.env = env ?? throw new ArgumentNullException(nameof(env));
        }
        public Space ObservationSpace => dispatcher.Call(() => env.ObservationSpace);
        public Space ActionSpace => dispatcher.Call(() => env.ActionSpace);
        public ResetResult Reset(long? seed, Metadata options) => dispatcher.Call(() => env.Reset(seed, options));
        public StepResult Step(Value action) => dispatcher.Call(() => env.Step(action));
    }
}
