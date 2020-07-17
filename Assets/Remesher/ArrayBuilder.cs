using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

sealed partial class Remesher
{
    struct Vertex
    {
        public float3 position;
        public float3 normal;
        public float4 tangent;
        public float2 texcoord;
    }

    struct Triangle
    {
        public Vertex v1;
        public Vertex v2;
        public Vertex v3;
    }

    static class ArrayBuilder
    {
        static NativeArray<T>
          TempMemory<T>(int length) where T : unmanaged => new NativeArray<T>
             (length, Allocator.Temp, NativeArrayOptions.UninitializedMemory);

        static NativeArray<T>
          TempJobMemory<T>(int length) where T : unmanaged => new NativeArray<T>
             (length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);

        public static NativeArray<uint> CreateIndexArray(Mesh source)
        {
            var count = (int)source.GetIndexCount(0);
            var buffer = TempMemory<uint>(count);
            for (var i = 0; i < count; i++) buffer[i] = (uint)i;
            return buffer;
        }

        #if NEVER_COMPILE_IT

        public static NativeArray<Vertex> CreateVertexArray(Mesh source)
        {
            using (var dataArray = Mesh.AcquireReadOnlyMeshData(source))
            {
                var data = dataArray[0];

                Debug.Assert(data.indexFormat == IndexFormat.UInt32);

                var vcount = data.vertexCount;
                var icount = data.GetSubMesh(0).indexCount;
                var src_idx = data.GetIndexData<uint>();

                using (var src_vtx = TempJobMemory<float3>(vcount))
                using (var src_uv0 = TempJobMemory<float2>(vcount))
                {
                    data.GetVertices(src_vtx.Reinterpret<Vector3>());
                    data.GetUVs(0, src_uv0.Reinterpret<Vector2>());

                    var out_vtx = TempJobMemory<Vertex>(icount);

                    for (var i = 0; i < icount; i += 3)
                    {
                        var i0 = (int)src_idx[i + 0];
                        var i1 = (int)src_idx[i + 1];
                        var i2 = (int)src_idx[i + 2];

                        var v0 = src_vtx[i0];
                        var v1 = src_vtx[i1];
                        var v2 = src_vtx[i2];

                        var t0 = src_uv0[i0];
                        var t1 = src_uv0[i1];
                        var t2 = src_uv0[i2];

                        var n = math.normalize(math.cross(v1 - v0, v2 - v0));
                        var t = math.float4(math.normalize(math.cross(n, math.float3(0, 1, 0))), 1);

                        out_vtx[i + 0] = new Vertex
                          { position = v0, normal = n, tangent = t, texcoord = t0 };
                        out_vtx[i + 1] = new Vertex
                          { position = v1, normal = n, tangent = t, texcoord = t1 };
                        out_vtx[i + 2] = new Vertex
                          { position = v2, normal = n, tangent = t, texcoord = t2 };
                    }

                    return out_vtx;
                }
            }
        }

        #else

        [Unity.Burst.BurstCompile(CompileSynchronously = true)]
        struct ReconstructionJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<uint> indices;
            [ReadOnly] public NativeArray<float3> vertex;
            [ReadOnly] public NativeArray<float2> texcoord;
            [WriteOnly] public NativeArray<Triangle> output;

            public void Execute(int i)
            {
                var i0 = (int)indices[i * 3 + 0];
                var i1 = (int)indices[i * 3 + 1];
                var i2 = (int)indices[i * 3 + 2];

                var v0 = vertex[i0];
                var v1 = vertex[i1];
                var v2 = vertex[i2];

                var t0 = texcoord[i0];
                var t1 = texcoord[i1];
                var t2 = texcoord[i2];

                var n = math.normalize(math.cross(v1 - v0, v2 - v0));
                var t = math.float4(math.normalize(math.cross(n, math.float3(0, 1, 0))), 1);

                output[i] = new Triangle {
                v1 = new Vertex
                  { position = v0, normal = n, tangent = t, texcoord = t0 },
                v2 = new Vertex
                  { position = v1, normal = n, tangent = t, texcoord = t1 },
                v3 = new Vertex
                  { position = v2, normal = n, tangent = t, texcoord = t2 }
                };

            }
        }

        public static NativeArray<Vertex> CreateVertexArray(Mesh source)
        {
            using (var dataArray = Mesh.AcquireReadOnlyMeshData(source))
            {
                var data = dataArray[0];

                Debug.Assert(data.indexFormat == IndexFormat.UInt32);

                var vcount = data.vertexCount;
                var icount = data.GetSubMesh(0).indexCount;
                var src_idx = data.GetIndexData<uint>();

                using (var src_vtx = TempJobMemory<float3>(vcount))
                using (var src_uv0 = TempJobMemory<float2>(vcount))
                {
                    data.GetVertices(src_vtx.Reinterpret<Vector3>());
                    data.GetUVs(0, src_uv0.Reinterpret<Vector2>());

                    var out_vtx = TempJobMemory<Vertex>(icount);

                    new ReconstructionJob {
                        indices = src_idx,
                        vertex = src_vtx,
                        texcoord = src_uv0,
                        output = out_vtx.Reinterpret<Triangle>(12 * 4)
                    }.Schedule(icount / 3, 64).Complete();

                    return out_vtx;
                }
            }
        }

        #endif
    }
}
