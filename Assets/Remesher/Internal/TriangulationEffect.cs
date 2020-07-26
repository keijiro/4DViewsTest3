using UnityEngine;
using UnityEngine.Rendering;
using Unity.Burst;
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
                    Out = out_vtx.Reinterpret<Triangle>(Vertex.StructSize) }
                  .Schedule(icount / 3, 64).Complete();

                return out_vtx;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
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

            // Indices
            var i1 = (int)Idx[i * 3 + 0];
            var i2 = (int)Idx[i * 3 + 1];
            var i3 = (int)Idx[i * 3 + 2];

            // Vertex positions with transformation
            var p1 = math.mul(Xfm, math.float4(Pos[i1], 1)).xyz;
            var p2 = math.mul(Xfm, math.float4(Pos[i2], 1)).xyz;
            var p3 = math.mul(Xfm, math.float4(Pos[i3], 1)).xyz;

            // Triangle centroid
            var pc = (p1 + p2 + p3) / 3;

            // Effect select
            var sel = hash.Float(8394) < 0.1f;

            // Effect parameter
            var eff = math.mul(Eff, math.float4(pc, 1)).z;
            eff = math.saturate(eff + hash.Float(0, 0.1f, 2058));

            // Deformation parameter
            var mod = (1 - math.cos(eff * math.PI * 4)) / 2;

            // Triangle scaling
            if (sel)
            {
                var scale = math.pow(hash.Float(84792), 8);
                scale = 1 + mod * math.lerp(5, 25, scale);
                p1 = math.lerp(pc, p1, scale);
                p2 = math.lerp(pc, p2, scale);
                p3 = math.lerp(pc, p3, scale);
            }

            // Normal/Tangent
            var nrm = MathUtil.UnitOrtho(p2 - p1, p3 - p1);
            var tan = MathUtil.AdHocTangent(nrm);

            // UV coordinates
            var mat = (eff > 0.25f && eff < 0.75f) ? 1.0f : 0.0f;
            var emm = (sel ? math.pow(mod, 20) * 2 : 0) - mat;
            var uv1 = math.float4(UV0[i1], mat, math.clamp(emm, -1, 1));
            var uv2 = math.float4(UV0[i2], uv1.zw);
            var uv3 = math.float4(UV0[i3], uv1.zw);

            // Output
            Out[i] = new Triangle(new Vertex(p1, nrm, tan, uv1),
                                  new Vertex(p2, nrm, tan, uv2),
                                  new Vertex(p3, nrm, tan, uv3));
        }
    }
}

}
