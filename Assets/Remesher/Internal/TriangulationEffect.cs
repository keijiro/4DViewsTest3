using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Remesher { 

static class TriangulationEffect
{
    public static NativeArray<Vertex>
      Build(Mesh source, Transform transform, Transform effector)
    {
        using (var dataArray = Mesh.AcquireReadOnlyMeshData(source))
        {
            var data = dataArray[0];

            // Vertex/index count
            var vcount = data.vertexCount;
            var icount = data.GetSubMesh(0).indexCount;

            // Source index array
            Debug.Assert(data.indexFormat == IndexFormat.UInt32);
            var src_idx = data.GetIndexData<uint>();

            // Read buffer allocation
            using (var src_pos = MemoryUtil.TempJobArray<float3>(vcount))
            using (var src_uv0 = MemoryUtil.TempJobArray<float2>(vcount))
            {
                // Retrieve vertex attribute arrays.
                data.GetVertices(src_pos.Reinterpret<Vector3>());
                data.GetUVs(0, src_uv0.Reinterpret<Vector2>());

                // Output buffer
                var out_vtx = MemoryUtil.TempJobArray<Vertex>(icount);

                // Invoke and wait the array generator job.
                new VertexArrayJob
                  { Idx = src_idx, Pos = src_pos, UV0 = src_uv0,
                    Xfm = transform.localToWorldMatrix,
                    Eff = effector.worldToLocalMatrix,
                    Out = out_vtx.Reinterpret<Triangle>(12 * 4) }
                  .Schedule(icount / 3, 64).Complete();

                return out_vtx;
            }
        }
    }

    [Unity.Burst.BurstCompile(CompileSynchronously = true)]
    struct VertexArrayJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> Idx;
        [ReadOnly] public NativeArray<float3> Pos;
        [ReadOnly] public NativeArray<float2> UV0;

        public float4x4 Xfm;
        public float4x4 Eff;

        [WriteOnly] public NativeArray<Triangle> Out;

        public void Execute(int i)
        {
            var hash = new Klak.Math.XXHash((uint)i);

            var i0 = (int)Idx[i * 3 + 0];
            var i1 = (int)Idx[i * 3 + 1];
            var i2 = (int)Idx[i * 3 + 2];

            var p0 = math.mul(Xfm, math.float4(Pos[i0], 1)).xyz;
            var p1 = math.mul(Xfm, math.float4(Pos[i1], 1)).xyz;
            var p2 = math.mul(Xfm, math.float4(Pos[i2], 1)).xyz;

            var uv0 = UV0[i0];
            var uv1 = UV0[i1];
            var uv2 = UV0[i2];

            var nrm = math.normalize(math.cross(p1 - p0, p2 - p0));
            var tan = math.float4(math.normalize(
              math.cross(nrm, math.float3(0, 1, 0))), 1);

            var pc = (p0 + p1 + p2) / 3;

            var mod = math.saturate(math.mul(Eff, math.float4(pc, 1)).z);

            var sel = hash.Float(0) < 0.1f;

            mod = (sel ?
             (math.smoothstep(0, 0.5f, mod) - math.smoothstep(0.5f, 1, mod)) * 20 : 0) + 1 - mod;

            p0 = math.lerp(pc, p0, mod);
            p1 = math.lerp(pc, p1, mod);
            p2 = math.lerp(pc, p2, mod);

            Out[i] = new Triangle(new Vertex(p0, nrm, tan, uv0),
                                  new Vertex(p1, nrm, tan, uv1),
                                  new Vertex(p2, nrm, tan, uv2));
        }
    }
}

}
