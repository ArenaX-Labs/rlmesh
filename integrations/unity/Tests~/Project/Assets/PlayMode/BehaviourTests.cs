using System.Collections;
using NUnit.Framework;
using RLMesh.Unity.Samples;
using UnityEngine;
using UnityEngine.TestTools;

namespace RLMesh.Unity.Verification
{
    public sealed class BehaviourTests
    {
        [UnityTest]
        public IEnumerator ExampleServesAndReturnsAnOffset()
        {
            var host = new GameObject("reach target test");
            try
            {
                var env = host.AddComponent<RLMeshReachTargetBehaviour>();
                env.Address = "tcp://127.0.0.1:0";
                yield return null;
                Assert.IsNotEmpty(env.GetServedAddress());
                IEnv authored = env;
                var reset = authored.Reset(42, null);
                using (var tensor = reset.Observation.AsTensor()) Assert.AreEqual(DType.Float32, tensor.Dtype);
                var result = authored.Step(1);
                Assert.LessOrEqual(result.Reward, 0);
                string address = env.GetServedAddress();
                env.Address = address; // Re-enable must await the old bind's drain.
                env.StopServing();
                env.enabled = false;
                yield return null;
                Assert.IsNull(env.GetServedAddress());
                env.enabled = true;
                float deadline = Time.realtimeSinceStartup + 5;
                while (env.GetServedAddress() == null)
                {
                    Assert.Less(Time.realtimeSinceStartup, deadline);
                    yield return null;
                }
                Assert.AreEqual(address, env.GetServedAddress());
                Assert.IsTrue(Application.runInBackground);
            }
            finally { Object.Destroy(host); }
        }
    }
}
