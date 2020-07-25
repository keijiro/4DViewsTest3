using UnityEngine;
using UnityEngine.Rendering;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using Klak.Math;

namespace Remesher { 

static class VoxelizationEffect
{
    #region Data structure

    public struct Element
    {
        public float3 Vertex1, Vertex2, Vertex3;
        public float2 UV1, UV2, UV3;

        public Element(float3 v1, float3 v2, float3 v3,
                       float2 uv1, float2 uv2, float2 uv3)
        {
            Vertex1 = v1; Vertex2 = v2; Vertex3 = v3;
            UV1 = uv1; UV2 = uv2; UV3 = uv3;
        }
    }

    public struct VoxelOutput
    {
        public Triangle Face1a, Face1b;
        public Triangle Face2a, Face2b;
        public Triangle Face3a, Face3b;
        public Triangle Face4a, Face4b;
        public Triangle Face5a, Face5b;
        public Triangle Face6a, Face6b;

        public VoxelOutput
          (in Vertex v1, in Vertex v2, in Vertex v3, in Vertex v4,
           in Vertex v5, in Vertex v6, in Vertex v7, in Vertex v8)
        {
            Face1a = new Triangle(v1, v2, v3);
            Face1b = new Triangle(v1, v2, v3);

            Face2a = new Triangle(v1, v2, v3);
            Face2b = new Triangle(v1, v2, v3);

            Face3a = new Triangle(v1, v2, v3);
            Face3b = new Triangle(v1, v2, v3);

            Face4a = new Triangle(v1, v2, v3);
            Face4b = new Triangle(v1, v2, v3);

            Face5a = new Triangle(v1, v2, v3);
            Face5b = new Triangle(v1, v2, v3);

            Face6a = new Triangle(v1, v2, v3);
            Face6b = new Triangle(v1, v2, v3);
        }
    }

    #endregion

    #region Element array initializer

    const int SourcePerVoxel = 10;

    public static (NativeArray<Element> voxels, NativeArray<Element> fragments)
      Initialize(Mesh sourceMesh, Transform transform)
    {
        using (var dataArray = Mesh.AcquireReadOnlyMeshData(sourceMesh))
        {
            var data = dataArray[0];

            // Vertex/index count
            var vtxCount = data.vertexCount;
            var idxCount = data.GetSubMesh(0).indexCount;

            // Source/voxel/fragment count
            var srcCount = idxCount / 3;
            var vxlCount = srcCount / SourcePerVoxel;
            var frgCount = srcCount - vxlCount;

            // Source index array
            Debug.Assert(data.indexFormat == IndexFormat.UInt32);
            var srcIdx = data.GetIndexData<uint>();

            // Read buffers allocation
            using (var srcPos = MemoryUtil.TempJobArray<float3>(vtxCount))
            using (var srcUV0 = MemoryUtil.TempJobArray<float2>(vtxCount))
            {
                // Retrieve vertex attribute arrays.
                data.GetVertices(srcPos.Reinterpret<Vector3>());
                data.GetUVs(0, srcUV0.Reinterpret<Vector2>());

                // Output buffer
                var outVxl = MemoryUtil.Array<Element>(vxlCount);
                var outFrg = MemoryUtil.Array<Element>(frgCount);

                // Invoke and wait the initializer jobs.
                var xform = transform.localToWorldMatrix;

                new InitializationJob
                  { SrcIdx = srcIdx, SrcPos = srcPos, SrcUV0 = srcUV0,
                    Transform = xform, IsVoxel = true, Output = outVxl }
                  .Schedule(vxlCount, 64).Complete();

                new InitializationJob
                  { SrcIdx = srcIdx, SrcPos = srcPos, SrcUV0 = srcUV0,
                    Transform = xform, IsVoxel = false, Output = outFrg }
                  .Schedule(frgCount, 64).Complete();

                return (outVxl, outFrg);
            }
        }
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct InitializationJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<uint> SrcIdx;
        [ReadOnly] public NativeArray<float3> SrcPos;
        [ReadOnly] public NativeArray<float2> SrcUV0;

        public bool IsVoxel;
        public float4x4 Transform;

        [WriteOnly] public NativeArray<Element> Output;

        // Index shuffle for voxel elements
        int SelectVoxelSource(int index)
          => index * SourcePerVoxel;

        // Index shuffle for fragment elements
        int SelectFragmentSource(int index)
          => index / (SourcePerVoxel - 1) * SourcePerVoxel +
             index % (SourcePerVoxel - 1) + 1;

        public void Execute(int i)
        {
            // Source index shuffling
            var si = IsVoxel ? SelectVoxelSource(i) : SelectFragmentSource(i);

            // Vertex indices
            var i1 = (int)SrcIdx[si * 3 + 0];
            var i2 = (int)SrcIdx[si * 3 + 1];
            var i3 = (int)SrcIdx[si * 3 + 2];

            // Output
            Output[i] = new Element(
              math.mul(Transform, math.float4(SrcPos[i1], 1)).xyz,
              math.mul(Transform, math.float4(SrcPos[i2], 1)).xyz,
              math.mul(Transform, math.float4(SrcPos[i3], 1)).xyz,
              SrcUV0[i1],
              SrcUV0[i2],
              SrcUV0[i3]
            );
        }
    }

    #endregion

    #region Vertex array builder

    public static NativeArray<Vertex> Build
      (NativeArray<Element> voxels, NativeArray<Element> fragments,
       Transform effector)
    {
        // Triangle count
        var triCount = voxels.Length * 12 + fragments.Length;

        // Output buffer
        var outVtx = MemoryUtil.TempJobArray<Vertex>(triCount * 3);
        var outVxl = outVtx.GetSubArray(0, voxels.Length * 12 * 3);
        var outFrg = outVtx.GetSubArray(outVxl.Length, fragments.Length * 3);

        // Invoke and wait for the builder job.
        var effMtx = (float4x4)effector.worldToLocalMatrix;

        new VoxelBuildJob
          { Elements = voxels, Effector = effMtx,
            Output = outVxl.Reinterpret<VoxelOutput>(Vertex.StructSize) }
          .Schedule(voxels.Length, 64).Complete();

        new FragmentBuildJob
          { Elements = fragments, Effector = effMtx,
            Output = outFrg.Reinterpret<Triangle>(Vertex.StructSize) }
          .Schedule(fragments.Length, 64).Complete();

        return outVtx;
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct VoxelBuildJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Element> Elements;

        public float4x4 Effector;

        [WriteOnly] public NativeArray<VoxelOutput> Output;

        public void Execute(int i)
        {
            var e = Elements[i];

            var p1 = e.Vertex1;
            var p2 = e.Vertex2;
            var p3 = e.Vertex3;

            var uv1 = math.float4(e.UV1, 0, 0);
            var uv2 = math.float4(e.UV2, 0, 0);
            var uv3 = math.float4(e.UV3, 0, 0);

            var nrm = MathUtil.UnitOrtho(p2 - p1, p3 - p1);
            var tan = MathUtil.AdHocTangent(nrm);

            var v1 = new Vertex(p1, nrm, tan, uv1);
            var v2 = new Vertex(p2, nrm, tan, uv2);
            var v3 = new Vertex(p3, nrm, tan, uv3);

            Output[i] = new VoxelOutput(v1, v2, v3, v1, v1, v2, v3, v1);
        }
    }

    [BurstCompile(CompileSynchronously = true,
      FloatMode = FloatMode.Fast, FloatPrecision = FloatPrecision.Low)]
    struct FragmentBuildJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<Element> Elements;

        public float4x4 Effector;

        [WriteOnly] public NativeArray<Triangle> Output;

        public void Execute(int i)
        {
            var e = Elements[i];

            var p1 = e.Vertex1;
            var p2 = e.Vertex2;
            var p3 = e.Vertex3;

            var uv1 = math.float4(e.UV1, 0, 0);
            var uv2 = math.float4(e.UV2, 0, 0);
            var uv3 = math.float4(e.UV3, 0, 0);

            var nrm = MathUtil.UnitOrtho(p2 - p1, p3 - p1);
            var tan = MathUtil.AdHocTangent(nrm);

            Output[i] = new Triangle(new Vertex(p1, nrm, tan, uv1),
                                     new Vertex(p2, nrm, tan, uv2),
                                     new Vertex(p3, nrm, tan, uv3));
        }
    }

    #endregion
}

}
