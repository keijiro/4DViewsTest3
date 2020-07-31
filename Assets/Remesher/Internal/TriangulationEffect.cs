using UnityEngine;
using UnityEngine.Rendering;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

namespace Remesher { 

static class TriangulationEffect
{
    public static NativeArray<Vertex> Build
      (in TriangulationConfig config, Mesh source,
       Transform transform, Transform effector)
    {
        using (var dataArray = Mesh.AcquireReadOnlyMeshData(source))
        {
            var data = dataArray[0];

            // Vertex/index count
            var vcount = data.vertexCount;
            var icount = data.GetSubMesh(0).indexCount;

            // Source index array
            Debug.Assert(data.indexFormat == IndexFormat.UInt32);
            var idx = data.GetIndexData<uint>();

            // Read buffer allocation
            using (var vtx = MemoryUtil.TempJobArray<float3>(vcount))
            using (var uvs = MemoryUtil.TempJobArray<float2>(vcount))
            {
                // Retrieve vertex attribute arrays.
                data.GetVertices(vtx.Reinterpret<Vector3>());
                data.GetUVs(0, uvs.Reinterpret<Vector2>());

                // Output buffer
                var output = MemoryUtil.TempJobArray<Vertex>(icount);

                // Invoke and wait the array generator job.
                new VertexArrayJob
                  { Config = config, Indices = idx, Vertices = vtx, UVs = uvs,
                    Transform = transform.localToWorldMatrix,
                    Effector = effector.worldToLocalMatrix,
                    Output = output.Reinterpret<Triangle>(Vertex.StructSize) }
                  .Schedule(icount / 3, 64).Complete();

                return output;
            }
        }
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct VertexArrayJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> Indices;
        [ReadOnly] public NativeArray<float3> Vertices;
        [ReadOnly] public NativeArray<float2> UVs;

        public TriangulationConfig Config;
        public float4x4 Transform;
        public float4x4 Effector;

        [WriteOnly] public NativeArray<Triangle> Output;

        public void Execute(int i)
        {
            var hash = new Klak.Math.XXHash((uint)i);

            // Indices
            var i1 = (int)Indices[i * 3 + 0];
            var i2 = (int)Indices[i * 3 + 1];
            var i3 = (int)Indices[i * 3 + 2];

            // Vertex positions with transformation
            var v1 = MathUtil.Transform(Transform, Vertices[i1]);
            var v2 = MathUtil.Transform(Transform, Vertices[i2]);
            var v3 = MathUtil.Transform(Transform, Vertices[i3]);

            // Source position (triangle centroid)
            var p = (v1 + v2 + v3) / 3;

            // Effect select
            var sel = hash.Float(8394) < Config.Probability;

            // Effect parameter
            var eff = MathUtil.Transform(Effector, p).z + 0.5f;
            eff += hash.Float(-0.5f, 0.5f, 2058) * Config.Softness;
            eff = math.saturate(eff);

            // Deformation parameter
            var mod = eff * 2 * math.PI * (Config.EffectType + 1);
            mod = (1 - math.cos(mod)) / 2;

            // Triangle scaling
            if (sel)
            {
                // The longest edge length
                var edge = math.max(math.length(v2 - v1), math.length(v3 - v1));

                // Scaling factor
                var sp = Config.ScaleParams;
                var scale = math.pow(hash.Float(84792), sp.z);
                scale = 1 + mod * math.lerp(sp.x, sp.y, scale) / edge;

                v1 = math.lerp(p, v1, scale);
                v2 = math.lerp(p, v2, scale);
                v3 = math.lerp(p, v3, scale);
            }

            // Normal/Tangent
            var nrm = MathUtil.UnitOrtho(v2 - v1, v3 - v1);
            var tan = MathUtil.AdHocTangent(nrm);

            // UV coordinates
            var mat = (eff > 0.25f && eff < 0.75f) ? Config.EffectType : 0;
            var emm = (sel ? math.pow(mod, 20) * 2 : 0) - mat;
            var uv1 = math.float4(UVs[i1], mat, math.clamp(emm, -1, 1));
            var uv2 = math.float4(UVs[i2], uv1.zw);
            var uv3 = math.float4(UVs[i3], uv1.zw);

            // Barycentric coordinates
            var bc1 = math.float4(1, 0, 0, 0);
            var bc2 = math.float4(0, 1, 0, 0);
            var bc3 = math.float4(0, 0, 1, 0);

            // Output
            Output[i] = new Triangle(new Vertex(v1, nrm, tan, bc1, uv1),
                                     new Vertex(v2, nrm, tan, bc2, uv2),
                                     new Vertex(v3, nrm, tan, bc3, uv3));
        }
    }
}

}
