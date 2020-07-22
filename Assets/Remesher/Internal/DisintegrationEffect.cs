using UnityEngine;
using UnityEngine.Rendering;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Klak.Math;

namespace Remesher { 

static class DisintegrationEffect
{
    #region Data structure

    public struct Source
    {
        public float3 Vertex1; public float2 UV1;
        public float3 Vertex2; public float2 UV2;
        public float3 Vertex3; public float2 UV3;
        public float3 Normal;
        public float4 Tangent;

        public Source
          (float3 vertex1, float2 uv1,
           float3 vertex2, float2 uv2,
           float3 vertex3, float2 uv3,
           float3 normal, float4 tangent)
        {
            Vertex1 = vertex1; UV1 = uv1;
            Vertex2 = vertex2; UV2 = uv2;
            Vertex3 = vertex3; UV3 = uv3;
            Normal = normal;
            Tangent = tangent;
        }
    }

    public struct Fragment
    {
        public float3 Position;
        public float3 Velocity;
        public float Morph;
        public Source Source;

        public Fragment(float3 position, float3 velocity, in Source source)
        {
            Position = position;
            Velocity = velocity;
            Morph = 0;
            Source = source;
        }
    }

    public struct TrianglePair
    {
        public Triangle Triangle1;
        public Triangle Triangle2;

        public TrianglePair(in Vertex v1, in Vertex v2, in Vertex v3,
                            in Vertex v4, in Vertex v5, in Vertex v6)
        {
            Triangle1 = new Triangle(v1, v2, v3);
            Triangle2 = new Triangle(v4, v5, v6);
        }
    }

    #endregion

    #region Fragment array initializer

    public static NativeArray<Fragment> Initialize
      (Mesh sourceMesh, Transform transform)
    {
        using (var dataArray = Mesh.AcquireReadOnlyMeshData(sourceMesh))
        {
            var data = dataArray[0];

            // Vertex/index count
            var vcount = data.vertexCount;
            var icount = data.GetSubMesh(0).indexCount;

            // Source index array
            Debug.Assert(data.indexFormat == IndexFormat.UInt32);
            var src_idx = data.GetIndexData<uint>();

            // Read buffers allocation
            using (var src_pos = MemoryUtil.TempJobArray<float3>(vcount))
            using (var src_uv0 = MemoryUtil.TempJobArray<float2>(vcount))
            {
                // Retrieve vertex attribute arrays.
                data.GetVertices(src_pos.Reinterpret<Vector3>());
                data.GetUVs(0, src_uv0.Reinterpret<Vector2>());

                // Output buffer
                var out_frags = MemoryUtil.Array<Fragment>(icount / 3);

                // Invoke and wait the initializer job.
                new InitializationJob
                  { Idx = src_idx, Pos = src_pos, UV0 = src_uv0,
                    Xfm = transform.localToWorldMatrix, Out = out_frags }
                  .Schedule(icount / 3, 64).Complete();

                return out_frags;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct InitializationJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> Idx;
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<float2> UV0;

        public float4x4 Xfm;

        [WriteOnly] public NativeArray<Fragment> Out;

        public void Execute(int i)
        {
            var i1 = (int)Idx[i * 3 + 0];
            var i2 = (int)Idx[i * 3 + 1];
            var i3 = (int)Idx[i * 3 + 2];

            var v1 = math.mul(Xfm, math.float4(Pos[i1], 1)).xyz;
            var v2 = math.mul(Xfm, math.float4(Pos[i2], 1)).xyz;
            var v3 = math.mul(Xfm, math.float4(Pos[i3], 1)).xyz;

            var uv1 = UV0[i1];
            var uv2 = UV0[i2];
            var uv3 = UV0[i3];

            var nrm = math.normalize(math.cross(v2 - v1, v3 - v1));
            var tan = math.float4(math.normalize(
              math.cross(nrm, math.float3(0, 1, 0))), 1);

            var pc = (v1 + v2 + v3) / 3;
            v1 -= pc;
            v2 -= pc;
            v3 -= pc;

            var src = new Source(v1, uv1, v2, uv2, v3, uv3, nrm, tan);
            Out[i] = new Fragment(pc, float3.zero, src);
        }
    }

    #endregion

    #region Fragment array updater

    public static void Update
      (NativeArray<Fragment> frags, Transform effector, float delta)
    {
        new UpdateJob
          { Frags = frags, Eff = effector.worldToLocalMatrix, Dt = delta }.
          Schedule(frags.Length, 64).Complete();
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct UpdateJob : IJobParallelFor
    {
        public NativeArray<Fragment> Frags;
        public float4x4 Eff;
        public float Dt;

        float Influence(float3 position)
          => math.mul(Eff, math.float4(position, 1)).z;

        float3 Acceleration(float3 position)
        {
            float3 n1, n2;
            noise.snoise(position.xyz * +2.3f, out n1);
            noise.snoise(position.zxy * -2.3f, out n2);
            return math.cross(n1, n2) * 0.1f;
        }

        float3 Lift => math.float3(0, 0.1f, 0);

        public void Execute(int i)
        {
            var frag = Frags[i];
            var hash = new XXHash((uint)i);

            // Effector influence test
            if (Influence(frag.Position) < 0) return;

            // Newtonian motion
            frag.Position += frag.Velocity * Dt;

            // Morphing parameter
            frag.Morph += Dt;

            // Acceleration by noise field
            frag.Velocity += Acceleration(frag.Position) * Dt;

            // Random lift force
            frag.Velocity += Lift * hash.Float(1, 2, 7487) * Dt;

            // Linear drag
            frag.Velocity -= frag.Velocity * Dt;

            Frags[i] = frag;
        }
    }

    #endregion

    #region Vertex array builder

    public static NativeArray<Vertex> Build(NativeArray<Fragment> frags)
    {
        // Triangle count
        var tcount = frags.Length;

        // Output buffer
        var out_vtx = MemoryUtil.TempJobArray<Vertex>(tcount * 6);

        // Invoke and wait for the builder job.
        new BuildJob
          { Frags = frags, Output = out_vtx.Reinterpret<TrianglePair>(12 * 4) }
          .Schedule(tcount, 64).Complete();

        return out_vtx;
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct BuildJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Fragment> Frags;
        [WriteOnly] public NativeArray<TrianglePair> Output;

        static float3 UnitOrthogonal(float3 a, float3 b)
          => math.normalizesafe(math.cross(a, b));

        public void Execute(int i)
        {
            var frag = Frags[i];
            var hash = new XXHash((uint)i);
            ref var src = ref frag.Source;

            // Base position
            var pos = frag.Position;

            // Forward vector
            var fwd = math.normalizesafe(frag.Velocity);

            // Right/Left hand vector
            var rhv = UnitOrthogonal(math.float3(0, 1, 0), fwd);
            var lhv = -rhv;

            // Flapping frequency
            var freq = hash.Float(8, 14, 3741);

            // Flapping parameter
            var flap = math.sin(frag.Morph * freq) * 1.5f;

            // Flapping
            rhv = math.mul(quaternion.AxisAngle(fwd, +flap), rhv);
            lhv = math.mul(quaternion.AxisAngle(fwd, -flap), lhv);

            // Original triangle
            var vs1 = pos + src.Vertex1;
            var vs2 = pos + src.Vertex2;
            var vs3 = pos + src.Vertex3;

            // Wind size parameter
            var life = math.lerp(5, 7, math.pow(hash.Float(4783), 4));
            var size = 0.02f * math.saturate(life - frag.Morph);

            // Right wing
            var v1 = pos;
            var v2 = pos + ( fwd + rhv) * size;
            var v3 = pos + (-fwd + rhv) * size;

            // Left wing
            var v4 = pos;
            var v5 = pos + (-fwd + lhv) * size;
            var v6 = pos + ( fwd + lhv) * size;

            // Morphing parameter
            var morph = math.saturate(frag.Morph * 0.5f);

            // Triangle to wings morphing
            v1 = math.lerp(vs1, v1, morph);
            v2 = math.lerp(vs2, v2, morph);
            v3 = math.lerp(vs3, v3, morph);

            v4 = math.lerp(vs1, v4, morph);
            v5 = math.lerp(vs2, v5, morph);
            v6 = math.lerp(vs3, v6, morph);

            // Normal vectors
            var n1 = UnitOrthogonal(v2 - v1, v3 - v1);
            var n2 = UnitOrthogonal(v5 - v4, v6 - v4);

            // Output
            Output[i] = new TrianglePair
              (new Vertex(v1, n1, src.Tangent, src.UV1),
               new Vertex(v2, n1, src.Tangent, src.UV2),
               new Vertex(v3, n1, src.Tangent, src.UV3),
               new Vertex(v4, n2, src.Tangent, src.UV1),
               new Vertex(v5, n2, src.Tangent, src.UV2),
               new Vertex(v6, n2, src.Tangent, src.UV3));
        }
    }

    #endregion
}

}
