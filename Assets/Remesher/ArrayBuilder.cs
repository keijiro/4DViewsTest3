using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

//
// Nested classes/structs of Remesher (only internally used)
//

sealed partial class Remesher
{
    #region Internal structs

    struct Vertex
    {
        public float3 Position;
        public float3 Normal;
        public float4 Tangent;
        public float2 TexCoord;

        public Vertex(float3 position, float3 normal,
                      float4 tangent, float2 texCoord)
        {
            Position = position;
            Normal = normal;
            Tangent = tangent;
            TexCoord = texCoord;
        }
    }

    struct Triangle
    {
        public Vertex Vertex1;
        public Vertex Vertex2;
        public Vertex Vertex3;

        public Triangle(in Vertex v1, in Vertex v2, in Vertex v3)
        {
            Vertex1 = v1;
            Vertex2 = v2;
            Vertex3 = v3;
        }
    }

    #endregion

    static class ArrayBuilder
    {
        #region NativeArray allocator

        static NativeArray<T> TempMemory<T>(int length) where T : unmanaged
            => new NativeArray<T>(length, Allocator.Temp,
                                  NativeArrayOptions.UninitializedMemory);

        static NativeArray<T> TempJobMemory<T>(int length) where T : unmanaged
            => new NativeArray<T>(length, Allocator.TempJob,
                                  NativeArrayOptions.UninitializedMemory);

        #endregion

        #region Index array builder

        // Simply enumerates all the vertices.

        public static NativeArray<uint> CreateIndexArray(Mesh source)
        {
            var count = (int)source.GetIndexCount(0);
            var array = TempJobMemory<uint>(count);
            new IndexArrayJob { Output = array, Count = count }.Run();
            return array;
        }

        [Unity.Burst.BurstCompile(CompileSynchronously = true)]
        struct IndexArrayJob : IJob
        {
            [WriteOnly] public NativeArray<uint> Output;
            public int Count;

            public void Execute()
            {
                for (var i = 0; i < Count; i++) Output[i] = (uint)i;
            }
        }

        #endregion

        #region Vertex array builder

        // Retrieve the original vertices using the new mesh API.
        // Then reconstruct a mesh using a Parallel-For job.

        public static NativeArray<Vertex> CreateVertexArray(Mesh source)
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
                using (var src_pos = TempJobMemory<float3>(vcount))
                using (var src_uv0 = TempJobMemory<float2>(vcount))
                {
                    // Retrieve vertex attribute arrays.
                    data.GetVertices(src_pos.Reinterpret<Vector3>());
                    data.GetUVs(0, src_uv0.Reinterpret<Vector2>());

                    // Output buffer
                    var out_vtx = TempJobMemory<Vertex>(icount);

                    // Invoke and wait the array generator job.
                    new VertexArrayJob
                      { Idx = src_idx, Pos = src_pos, UV0 = src_uv0,
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
            [WriteOnly] public NativeArray<Triangle> Out;

            public void Execute(int i)
            {
                var i0 = (int)Idx[i * 3 + 0];
                var i1 = (int)Idx[i * 3 + 1];
                var i2 = (int)Idx[i * 3 + 2];

                var p0 = Pos[i0];
                var p1 = Pos[i1];
                var p2 = Pos[i2];

                var uv0 = UV0[i0];
                var uv1 = UV0[i1];
                var uv2 = UV0[i2];

                var nrm = math.normalize(math.cross(p1 - p0, p2 - p0));
                var tan = math.float4(math.normalize(
                  math.cross(nrm, math.float3(0, 1, 0))), 1);

                Out[i] = new Triangle(new Vertex(p0, nrm, tan, uv0),
                                      new Vertex(p1, nrm, tan, uv1),
                                      new Vertex(p2, nrm, tan, uv2));
            }
        }

        #endregion
    }
}
