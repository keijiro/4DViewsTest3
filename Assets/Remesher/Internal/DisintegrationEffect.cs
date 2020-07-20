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
        public float3 Position1; public float2 UV1;
        public float3 Position2; public float2 UV2;
        public float3 Position3; public float2 UV3;
        public float3 Normal;
        public float4 Tangent;

        public Source
          (float3 position1, float2 uv1,
           float3 position2, float2 uv2,
           float3 position3, float2 uv3,
           float3 normal, float4 tangent)
        {
            Position1 = position1; UV1 = uv1;
            Position2 = position2; UV2 = uv2;
            Position3 = position3; UV3 = uv3;
            Normal = normal;
            Tangent = tangent;
        }
    }

    public struct Fragment
    {
        public float3 Position;
        public float3 Velocity;
        public Source Source;

        public Fragment(float3 position, float3 velocity, in Source source)
        {
            Position = position;
            Velocity = velocity;
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

            var p1 = math.mul(Xfm, math.float4(Pos[i1], 1)).xyz;
            var p2 = math.mul(Xfm, math.float4(Pos[i2], 1)).xyz;
            var p3 = math.mul(Xfm, math.float4(Pos[i3], 1)).xyz;

            var uv1 = UV0[i1];
            var uv2 = UV0[i2];
            var uv3 = UV0[i3];

            var pc = (p1 + p2 + p3) / 3;

            var nrm = math.normalize(math.cross(p2 - p1, p3 - p1));
            var tan = math.float4(math.normalize(
              math.cross(nrm, math.float3(0, 1, 0))), 1);

            var src = new Source(p1, uv1, p2, uv2, p3, uv3, nrm, tan);
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

            frag.Source.Position1 += math.float3(0, Dt, 0);

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
            var src = Frags[i].Source;

            Output[i] = new Triangle
              (new Vertex(src.Position1, src.Normal, src.Tangent, src.UV1),
               new Vertex(src.Position2, src.Normal, src.Tangent, src.UV2),
               new Vertex(src.Position3, src.Normal, src.Tangent, src.UV3));
        }
    }

    #endregion
}

}
