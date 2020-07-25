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
          (float3 v1, float4 uv1, float3 v2, float4 uv2,
           float3 v3, float4 uv3, float3 v4, float4 uv4,
           float3 v5, float4 uv5, float3 v6, float4 uv6,
           float3 v7, float4 uv7, float3 v8, float4 uv8)
        {
            var uv = math.float4(0, 0, 1, 0);

            ConstructFace(v2, uv2, v1, uv1, v4, uv4, out Face1a);
            ConstructFace(v1, uv1, v3, uv3, v4, uv4, out Face1b);

            ConstructFace(v5, uv5, v6, uv6, v7, uv7, out Face2a);
            ConstructFace(v6, uv6, v8, uv8, v7, uv7, out Face2b);

            ConstructFace(v1, uv1, v5, uv5, v7, uv7, out Face3a);
            ConstructFace(v1, uv1, v7, uv7, v3, uv3, out Face3b);

            ConstructFace(v6, uv6, v2, uv2, v4, uv4, out Face4a);
            ConstructFace(v6, uv6, v4, uv4, v8, uv8, out Face4b);

            ConstructFace(v1, uv1, v2, uv2, v5, uv5, out Face5a);
            ConstructFace(v2, uv2, v6, uv6, v5, uv5, out Face5b);

            ConstructFace(v7, uv7, v8, uv8, v3, uv3, out Face6a);
            ConstructFace(v8, uv8, v4, uv4, v3, uv3, out Face6b);
        }

        static void ConstructFace(float3 v1, float4 uv1,
                                  float3 v2, float4 uv2,
                                  float3 v3, float4 uv3,
                                  out Triangle tri)
        {
            var nrm = MathUtil.UnitOrtho(v2 - v1, v3 - v1);
            var tan = MathUtil.AdHocTangent(nrm);
            tri = new Triangle(new Vertex(v1, nrm, tan, uv1),
                               new Vertex(v2, nrm, tan, uv2),
                               new Vertex(v3, nrm, tan, uv3));
        }
    }

    #endregion

    #region Element array initializer

    const int SourcePerVoxel = 20;

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
            var vxlCount = (srcCount + SourcePerVoxel - 1) / SourcePerVoxel;
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
            var hash = new XXHash((uint)i);

            // Vertices
            var p1 = e.Vertex1;
            var p2 = e.Vertex2;
            var p3 = e.Vertex3;

            // Centroid
            var pc = (p1 + p2 + p3) / 3;

            // Effect parameter
            var eff = -math.mul(Effector, math.float4(pc, 1)).z;
            eff -= hash.Float(0.4f, 3244);

            // Triangle vertices
            var triMod = 1 + math.smoothstep(0, 0.3f, eff) * 30;
            var vt1 = math.lerp(pc, p1, triMod);
            var vt2 = math.lerp(pc, p2, triMod);
            var vt3 = math.lerp(pc, p3, triMod);

            // Motion by noise
            float3 nm;
            noise.snoise(pc * 3.3f + math.float3(0, eff * 1.3f, 0), out nm);
            pc += nm * 0.005f * (1 - math.smoothstep(0.6f, 0.7f, eff));

            // Voxel travel
            var trv = math.smoothstep(0.7f, 1.3f, eff);
            pc.z -= trv * 2;

            // Voxel size
            var s1 = 1 + trv * 30;
            var s2 = 1 - trv;
            var size = 0.03f * math.float3(s2, s2, s1);

            // Voxel vertices
            var vc1 = pc + math.float3(-1, -1, -1) * size;
            var vc2 = pc + math.float3(+1, -1, -1) * size;
            var vc3 = pc + math.float3(-1, +1, -1) * size;
            var vc4 = pc + math.float3(+1, +1, -1) * size;
            var vc5 = pc + math.float3(-1, -1, +1) * size;
            var vc6 = pc + math.float3(+1, -1, +1) * size;
            var vc7 = pc + math.float3(-1, +1, +1) * size;
            var vc8 = pc + math.float3(+1, +1, +1) * size;

            // Triangle to voxel deformation
            var vxlMod = math.smoothstep(0.1f, 0.3f, eff);
            vc1 = math.lerp(vt1, vc1, vxlMod);
            vc2 = math.lerp(vt2, vc2, vxlMod);
            vc3 = math.lerp(vt3, vc3, vxlMod);
            vc4 = math.lerp(vt3, vc4, vxlMod);
            vc5 = math.lerp(vt1, vc5, vxlMod);
            vc6 = math.lerp(vt2, vc6, vxlMod);
            vc7 = math.lerp(vt3, vc7, vxlMod);
            vc8 = math.lerp(vt3, vc8, vxlMod);

            var em = math.saturate(eff * 10);
            var uv1 = math.float4(e.UV1, em, em - vxlMod + trv);
            var uv2 = math.float4(e.UV2, em, em - vxlMod + trv);
            var uv3 = math.float4(e.UV3, em, em - vxlMod + trv);

            Output[i] = new VoxelOutput
              (vc1, uv1, vc2, uv2, vc3, uv3, vc4, uv3,
               vc5, uv1, vc6, uv2, vc7, uv3, vc8, uv3);
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
            var hash = new XXHash((uint)i);

            // Vertices
            var p1 = e.Vertex1;
            var p2 = e.Vertex2;
            var p3 = e.Vertex3;

            // Centroid
            var pc = (p1 + p2 + p3) / 3;

            // Effect parameter
            var eff = -math.mul(Effector, math.float4(pc, 1)).z;
            eff -= hash.Float(0.4f, 3244);

            // Motion by noise
            float3 nm;
            noise.snoise(pc * 3.3f, out nm);
            pc += nm * 0.05f * math.smoothstep(0, 1, eff);

            // Simple shrink
            var mod = math.saturate(eff);
            p1 = math.lerp(p1, pc, mod);
            p2 = math.lerp(p2, pc, mod);
            p3 = math.lerp(p3, pc, mod);

            // Vertex attributes
            var em = math.saturate(eff * 10);
            var uv1 = math.float4(e.UV1, em, em);
            var uv2 = math.float4(e.UV2, em, em);
            var uv3 = math.float4(e.UV3, em, em);
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
