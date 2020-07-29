using UnityEngine;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Remesher {

static class LightStripController
{
    #region Data structure

    public struct Element
    {
        public float3 Position;
        public float3 Color;

        public Element(float3 position, float3 color)
        {
            Position = position;
            Color = color;
        }
    }

    #endregion

    #region Constant numbers

    const int VerticesPerRing = 6;
    const float RingWidth = 0.006f;

    #endregion

    #region Vertex functions

    public static float3 GetVertexPosition
      (in LightStripConfig config, float time)
      => math.float3(config.Radius * math.cos(config.Motion.x * time),
                     config.Height * math.sin(config.Motion.y * time),
                     config.Radius * math.sin(config.Motion.x * time));

    public static float3 GetVertexColor
      (in LightStripConfig config, int index, float time)
      => config.Gradient.EvaluateAsFloat3
        ((float)index / config.VertexCount + config.GradientScroll * time);

    #endregion

    #region Element array initializer

    public static NativeArray<Element> Initialize(in LightStripConfig config)
    {
        var output = MemoryUtil.Array<Element>(config.VertexCount);
        new InitializeJob { Config = config, Elements = output }.Run();
        return output;
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct InitializeJob : IJob
    {
        [ReadOnly] public LightStripConfig Config;

        public NativeArray<Element> Elements;

        public void Execute()
        {
            for (var i = 0; i < Elements.Length; i++)
            {
                var t = i / -30.0f;
                var p = GetVertexPosition(Config, t);
                var c = GetVertexColor(Config, i, t);
                Elements[i] = new Element(p, c);
            }
        }
    }

    #endregion

    #region Element array updater

    public static void Update
      (in LightStripConfig config,
       NativeArray<Element> elements, float time, float deltaTime)
    {
        new UpdateJob { Config = config, Elements = elements,
                        Time = time, DeltaTime = deltaTime }.Run();
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct UpdateJob : IJob
    {
        [ReadOnly] public LightStripConfig Config;

        public NativeArray<Element> Elements;
        public float Time, DeltaTime;

        public void Execute()
        {
            var ecount = Elements.Length;

            for (var i = 0; i < ecount - 1; i++)
            {
                var p = Elements[i + 1].Position;

                // Noise field reference point
                var np = p * Config.NoiseFrequency;
                np.y += Config.NoiseMotion * Time;

                // Advection by divergence-free noise field
                p += MathUtil.DFNoise(np) * DeltaTime * Config.NoiseAmplitude;

                // Coloring
                var c = GetVertexColor(Config, i, Time);

                Elements[i] = new Element(p, c);
            }

            // Head animation
            {
                var p = GetVertexPosition(Config, Time);
                var c = GetVertexColor(Config, ecount - 1, Time);
                Elements[ecount - 1] = new Element(p, c);
            }
        }
    }

    #endregion

    #region Vertex array builder

    public static NativeArray<Vertex>
      BuildVertexArray(NativeArray<Element> elements)
    {
        var vcount = elements.Length * VerticesPerRing;
        var output = MemoryUtil.TempJobArray<Vertex>(vcount);
        new VertexBuildJob { Elements = elements, Output = output }.Run();
        return output;
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct VertexBuildJob : IJob
    {
        [ReadOnly] public NativeArray<Element> Elements;

        [WriteOnly] public NativeArray<Vertex> Output;

        public void Execute()
        {
            var outIdx = 0;

            for (var i = 0; i < Elements.Length; i++)
            {
                // Current element
                var p = Elements[i].Position;
                var c = Elements[i].Color;

                // Previous/next element index
                var i_p = math.max(i - 1, 0);
                var i_n = math.min(i + 1, Elements.Length - 1);

                // Previous/next element position
                var p_p = Elements[i_p].Position;
                var p_n = Elements[i_n].Position;

                // Frenet-Serret frame
                var fs_t = math.normalizesafe(p_n - p_p);
                var fs_n = math.normalizesafe(p_n - p * 2 + p_p);
                var fs_b = math.normalizesafe(math.cross(fs_t, fs_n));
                fs_n = math.normalizesafe(math.cross(fs_b, fs_t));

                // Tangent/tex-coord
                var tan = math.float4(fs_t, 1);
                var tex= math.float4(c, 1);

                for (var j = 0; j < VerticesPerRing; j++)
                {
                    var theta = math.PI * 2 / VerticesPerRing * j;

                    // Normal
                    var n = fs_n * math.cos(theta) + fs_b * math.sin(theta);

                    // Vertex position
                    var v = p + n * RingWidth;

                    // Output
                    Output[outIdx++] = new Vertex(v, n, tan, tex);
                }
            }
        }
    }

    #endregion

    #region Index array builder

    public static NativeArray<uint>
      BuildIndexArray(NativeArray<Element> elements)
    {
        var ecount = elements.Length;
        var icount = (ecount - 1) * VerticesPerRing * 3 * 2;
        var output = MemoryUtil.TempJobArray<uint>(icount);
        new IndexBuildJob { ElementCount = ecount, Output = output }.Run();
        return output;
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct IndexBuildJob : IJob
    {
        public int ElementCount;

        [WriteOnly] public NativeArray<uint> Output;

        public void Execute()
        {
            var refIdx = 0u;
            var outIdx = 0;

            for (var i = 0u; i < ElementCount - 1; i++)
            {
                for (var j = 0u; j < VerticesPerRing; j++)
                {
                    var i0 = refIdx + j;
                    var i1 = refIdx + (j + 1) % VerticesPerRing;
                    var i2 = i0 + VerticesPerRing;
                    var i3 = i1 + VerticesPerRing;

                    Output[outIdx++] = i0;
                    Output[outIdx++] = i1;
                    Output[outIdx++] = i2;

                    Output[outIdx++] = i2;
                    Output[outIdx++] = i1;
                    Output[outIdx++] = i3;
                }

                refIdx += VerticesPerRing;
            }
        }
    }

    #endregion
}

}
