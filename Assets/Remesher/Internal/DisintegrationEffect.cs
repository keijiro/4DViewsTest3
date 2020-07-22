using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

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

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
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

    public static void Update(NativeArray<Fragment> frags, float delta)
    {
        new UpdateJob { Frags = frags, Dt = delta }.
          Schedule(frags.Length, 64).Complete();
    }

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct UpdateJob : IJobParallelFor
    {
        public NativeArray<Fragment> Frags;
        public float Dt;

        public void Execute(int i)
        {
            Fragment frag = Frags[i];

            frag.Position += frag.Velocity * Dt;
            frag.Morph += Dt;

            var np1 = frag.Position * 2.3f;
            var np2 = frag.Position.zxy * -2.3f;

            float3 n1, n2;
            noise.snoise(np1, out n1);
            noise.snoise(np2, out n2);

            frag.Velocity += math.cross(n1, n2) * Dt * 0.1f;
            frag.Velocity.y += 0.2f * Dt;
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
        var out_vtx = MemoryUtil.TempJobArray<Vertex>(tcount * 3);

        // Invoke and wait for the builder job.
        new BuildJob
          { Frags = frags, Output = out_vtx.Reinterpret<Triangle>(12 * 4) }
          .Schedule(tcount, 64).Complete();

        return out_vtx;
    }

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct BuildJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Fragment> Frags;
        [WriteOnly] public NativeArray<Triangle> Output;

        public void Execute(int i)
        {
            var frag = Frags[i];
            var src = Frags[i].Source;

            var param = 1 + math.saturate(frag.Morph);

            var p = frag.Position;
            var v1 = p + src.Vertex1 * param;
            var v2 = p + src.Vertex2 * param;
            var v3 = p + src.Vertex3 * param;

            Output[i] = new Triangle
              (new Vertex(v1, src.Normal, src.Tangent, src.UV1),
               new Vertex(v2, src.Normal, src.Tangent, src.UV2),
               new Vertex(v3, src.Normal, src.Tangent, src.UV3));
        }
    }

    #endregion
}

}
