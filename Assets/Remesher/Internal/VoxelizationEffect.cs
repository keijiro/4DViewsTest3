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

    // Effect element (voxel or fragment)
    public struct Element
    {
        public float3 Vertex1; public float2 UV1;
        public float3 Vertex2; public float2 UV2;
        public float3 Vertex3; public float2 UV3;

        public Element(float3 v1, float2 uv1,
                       float3 v2, float2 uv2,
                       float3 v3, float2 uv3)
        {
            Vertex1 = v1; UV1 = uv1;
            Vertex2 = v2; UV2 = uv2;
            Vertex3 = v3; UV3 = uv3;
        }
    }

    // Output struct for voxel elements (memory view for vertex array)
    public struct VoxelOutput
    {
        public Triangle _face1a, _face1b;
        public Triangle _face2a, _face2b;
        public Triangle _face3a, _face3b;
        public Triangle _face4a, _face4b;
        public Triangle _face5a, _face5b;
        public Triangle _face6a, _face6b;

        public VoxelOutput(float3 v1, float4 uv1, float3 v2, float4 uv2,
                           float3 v3, float4 uv3, float3 v4, float4 uv4,
                           float3 v5, float4 uv5, float3 v6, float4 uv6,
                           float3 v7, float4 uv7, float3 v8, float4 uv8)
        {
            ConstructFace(v2, uv2, v1, uv1, v4, uv4, out _face1a);
            ConstructFace(v1, uv1, v3, uv3, v4, uv4, out _face1b);

            ConstructFace(v5, uv5, v6, uv6, v7, uv7, out _face2a);
            ConstructFace(v6, uv6, v8, uv8, v7, uv7, out _face2b);

            ConstructFace(v1, uv1, v5, uv5, v7, uv7, out _face3a);
            ConstructFace(v1, uv1, v7, uv7, v3, uv3, out _face3b);

            ConstructFace(v6, uv6, v2, uv2, v4, uv4, out _face4a);
            ConstructFace(v6, uv6, v4, uv4, v8, uv8, out _face4b);

            ConstructFace(v1, uv1, v2, uv2, v5, uv5, out _face5a);
            ConstructFace(v2, uv2, v6, uv6, v5, uv5, out _face5b);

            ConstructFace(v7, uv7, v8, uv8, v3, uv3, out _face6a);
            ConstructFace(v8, uv8, v4, uv4, v3, uv3, out _face6b);
        }

        static void ConstructFace(float3 v1, float4 uv1,
                                  float3 v2, float4 uv2,
                                  float3 v3, float4 uv3,
                                  out Triangle face)
        {
            var normal = MathUtil.UnitOrtho(v2 - v1, v3 - v1);
            var tangent = MathUtil.AdHocTangent(normal);
            face = new Triangle(new Vertex(v1, normal, tangent, uv1),
                                new Vertex(v2, normal, tangent, uv2),
                                new Vertex(v3, normal, tangent, uv3));
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

            // Triangle/voxel/fragment count
            var triCount = idxCount / 3;
            var vxlCount = (triCount + SourcePerVoxel - 1) / SourcePerVoxel;
            var frgCount = triCount - vxlCount;

            // Source index array
            Debug.Assert(data.indexFormat == IndexFormat.UInt32);
            var idx = data.GetIndexData<uint>();

            // Read buffers allocation
            using (var vtx = MemoryUtil.TempJobArray<float3>(vtxCount))
            using (var uvs = MemoryUtil.TempJobArray<float2>(vtxCount))
            {
                // Retrieve vertex attribute arrays.
                data.GetVertices(vtx.Reinterpret<Vector3>());
                data.GetUVs(0, uvs.Reinterpret<Vector2>());

                // Output buffer
                var outVxl = MemoryUtil.Array<Element>(vxlCount);
                var outFrg = MemoryUtil.Array<Element>(frgCount);

                // Invoke and wait the initializer jobs.
                var xform = transform.localToWorldMatrix;

                new InitializationJob
                  { Indices = idx, Vertices = vtx, UVs = uvs,
                    Transform = xform, IsVoxel = true, Output = outVxl }
                  .Schedule(vxlCount, 64).Complete();

                new InitializationJob
                  { Indices = idx, Vertices = vtx, UVs = uvs,
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
        [ReadOnly] public NativeArray<uint> Indices;
        [ReadOnly] public NativeArray<float3> Vertices;
        [ReadOnly] public NativeArray<float2> UVs;

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
            var i1 = (int)Indices[si * 3 + 0];
            var i2 = (int)Indices[si * 3 + 1];
            var i3 = (int)Indices[si * 3 + 2];

            // Output
            Output[i] = new Element
              (MathUtil.Transform(Transform, Vertices[i1]), UVs[i1],
               MathUtil.Transform(Transform, Vertices[i2]), UVs[i2],
               MathUtil.Transform(Transform, Vertices[i3]), UVs[i3]);
        }
    }

    #endregion

    #region Vertex array builder

    public static NativeArray<Vertex>
      Build(NativeArray<Element> voxels, NativeArray<Element> fragments,
            Transform effector)
    {
        // Output triangle count
        var triCount = voxels.Length * 12 + fragments.Length;

        // Output buffer
        var outVtx = MemoryUtil.TempJobArray<Vertex>(triCount * 3);

        // Number of vertices in the voxel array
        var vtxInVxl = voxels.Length * 12 * 3;

        // Sub-arrays as structured output memory views
        var outVxl = outVtx.GetSubArray(0, vtxInVxl).
                     Reinterpret<VoxelOutput>(Vertex.StructSize);

        var outFrg = outVtx.GetSubArray(vtxInVxl, fragments.Length * 3).
                     Reinterpret<Triangle>(Vertex.StructSize);

        // Effector matrix
        var effMtx = (float4x4)effector.worldToLocalMatrix;

        // Invoke and wait for the builder job.
        new VoxelBuildJob
          { Elements = voxels, Effector = effMtx, Output = outVxl }
          .Schedule(voxels.Length, 64).Complete();

        new FragmentBuildJob
          { Elements = fragments, Effector = effMtx, Output = outFrg }
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

            // Source triangle vertices
            var v1 = e.Vertex1;
            var v2 = e.Vertex2;
            var v3 = e.Vertex3;

            // Source position (triangle centroid)
            var p = (v1 + v2 + v3) / 3;

            // Effect parameter
            var eff = -math.mul(Effector, math.float4(p, 1)).z;
            eff -= hash.Float(0.4f, 3244); // Random distribution

            // Triangle deformation (expand very fast)
            var tmod = 1 + math.smoothstep(0, 0.3f, eff) * 30;
            v1 = math.lerp(p, v1, tmod);
            v2 = math.lerp(p, v2, tmod);
            v3 = math.lerp(p, v3, tmod);

            // Voxel motion by noise (stops before travelling)
            var nmod = 1 - math.smoothstep(0.6f, 0.7f, eff);
            var nfp = p * 3.3f + math.float3(0, eff * 0.4f, 0);
            p += MathUtil.DFNoise(nfp) * 0.004f * nmod;

            // Voxel moving out
            var mout = math.smoothstep(0.7f, 1.3f, eff);
            p.z -= mout * 2;

            // Voxel scaling
            var vs = 0.03f * math.float2(1 - mout, 1 + mout * 30).xxy;

            // Voxel vertices
            var vv1 = p + math.float3(-1, -1, -1) * vs;
            var vv2 = p + math.float3(+1, -1, -1) * vs;
            var vv3 = p + math.float3(-1, +1, -1) * vs;
            var vv4 = p + math.float3(+1, +1, -1) * vs;
            var vv5 = p + math.float3(-1, -1, +1) * vs;
            var vv6 = p + math.float3(+1, -1, +1) * vs;
            var vv7 = p + math.float3(-1, +1, +1) * vs;
            var vv8 = p + math.float3(+1, +1, +1) * vs;

            // Triangle to voxel deformation
            var vmod = math.smoothstep(0.1f, 0.3f, eff);
            vv1 = math.lerp(v1, vv1, vmod);
            vv2 = math.lerp(v2, vv2, vmod);
            vv3 = math.lerp(v3, vv3, vmod);
            vv4 = math.lerp(v3, vv4, vmod);
            vv5 = math.lerp(v1, vv5, vmod);
            vv6 = math.lerp(v2, vv6, vmod);
            vv7 = math.lerp(v3, vv7, vmod);
            vv8 = math.lerp(v3, vv8, vmod);

            // UV and material parameters
            var mat = math.saturate(eff * 10);
            var emm = mat - vmod +
              math.smoothstep(0.6f, 0.7f, eff) * hash.Float(1.1f, 2.1f, 1893);
            var uv1 = math.float4(e.UV1, mat, emm);
            var uv2 = math.float4(e.UV2, mat, emm);
            var uv3 = math.float4(e.UV3, mat, emm);

            // Output
            Output[i] = new VoxelOutput
              (vv1, uv1, vv2, uv2, vv3, uv3, vv4, uv3,
               vv5, uv1, vv6, uv2, vv7, uv3, vv8, uv3);
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

            // Source triangle vertices
            var v1 = e.Vertex1;
            var v2 = e.Vertex2;
            var v3 = e.Vertex3;

            // Source position (triangle centroid)
            var p = (v1 + v2 + v3) / 3;

            // Effect parameter
            var eff = -math.mul(Effector, math.float4(p, 1)).z;
            eff -= hash.Float(0.4f, 4773); // Random distribution

            // Triangle shrink
            var tmod = math.saturate(eff);
            v1 = math.lerp(v1, p, tmod);
            v2 = math.lerp(v2, p, tmod);
            v3 = math.lerp(v3, p, tmod);

            // Motion by noise
            var offs = MathUtil.DFNoise(p * 3.3f) * 0.02f;
            offs *= math.smoothstep(0, 1, eff);

            v1 += offs;
            v2 += offs;
            v3 += offs;

            // Normal/tangent
            var nrm = MathUtil.UnitOrtho(v2 - v1, v3 - v1);
            var tan = MathUtil.AdHocTangent(nrm);

            // UV and material parameters
            var em = math.saturate(eff * 10);
            var uv1 = math.float4(e.UV1, em, em);
            var uv2 = math.float4(e.UV2, em, em);
            var uv3 = math.float4(e.UV3, em, em);

            // Output
            Output[i] = new Triangle(new Vertex(v1, nrm, tan, uv1),
                                     new Vertex(v2, nrm, tan, uv2),
                                     new Vertex(v3, nrm, tan, uv3));
        }
    }

    #endregion
}

}
