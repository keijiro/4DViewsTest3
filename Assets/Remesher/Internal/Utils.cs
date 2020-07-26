using UnityEngine;
using UnityEngine.Rendering;
using Unity.Collections;
using Unity.Mathematics;

namespace Remesher {

//
// Common-use utility classes
//

#region Math utilities

static class MathUtil
{
    public static float3 UnitOrtho(float3 a, float3 b)
      => math.normalizesafe(math.cross(a, b));

    public static float4 AdHocTangent(float3 normal)
      => math.float4(UnitOrtho(normal, math.float3(0.01f, 1, 0.01f)), 1);

    public static float3 Transform(in float4x4 matrix, float3 point)
      => math.mul(matrix, math.float4(point, 1)).xyz;

    public static float3 DFNoise(float3 position)
    {
        float3 n1, n2;
        noise.snoise(+position.xyz + math.float3(3, 4, 5), out n1);
        noise.snoise(-position.zxy - math.float3(2, 3, 4), out n2);
        return math.cross(n1, n2);
    }
}

#endregion

#region UnityEngine.Object management utilities

static class ObjectUtil
{
    public static void Destroy(Object o)
    {
        if (o == null) return;
        if (Application.isPlaying)
            Object.Destroy(o);
        else
            Object.DestroyImmediate(o);
    }
}

#endregion

#region Memory management utilities (mainly for NativeArray)

static class MemoryUtil
{
    public static NativeArray<T> Array<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.Persistent,
                              NativeArrayOptions.UninitializedMemory);

    public static NativeArray<T> TempArray<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.Temp,
                              NativeArrayOptions.UninitializedMemory);

    public static NativeArray<T> TempJobArray<T>(int length) where T : struct
        => new NativeArray<T>(length, Allocator.TempJob,
                              NativeArrayOptions.UninitializedMemory);
}

#endregion

#region Utilities for geometric objects

static class GeomUtil
{
    public static Bounds TransformBounds(Bounds bounds, Transform transform)
    {
        // Very crude approximation
        var center = transform.TransformPoint(bounds.center);
        var size = transform.TransformVector(bounds.size);
        var maxs = Mathf.Max(Mathf.Max(size.x, size.y), size.z);
        return new Bounds(center, Vector3.one * maxs);
    }
}

#endregion

#region Mesh object management utilities

static class MeshUtil
{
    public static Mesh SetupWithMeshFilter(GameObject parent)
    {
        var mesh = new Mesh();
        mesh.hideFlags = HideFlags.DontSave;
        mesh.MarkDynamic();

        var mf = parent.GetComponent<MeshFilter>();
        if (mf == null)
        {
            mf = parent.AddComponent<MeshFilter>();
            mf.hideFlags = HideFlags.NotEditable | HideFlags.DontSave;
        }

        mf.sharedMesh = mesh;

        return mesh;
    }

    public static void UpdateWithVertexIndexArrays
      (Mesh mesh, NativeArray<Vertex> vertices, NativeArray<uint> indices)
    {
        mesh.Clear();

        var vcount = vertices.Length;

        mesh.SetVertexBufferParams(
          vcount,
          new VertexAttributeDescriptor
            (VertexAttribute.Position  , VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Normal    , VertexAttributeFormat.Float32, 3),
          new VertexAttributeDescriptor
            (VertexAttribute.Tangent   , VertexAttributeFormat.Float32, 4),
          new VertexAttributeDescriptor
            (VertexAttribute.TexCoord0 , VertexAttributeFormat.Float32, 2),
          new VertexAttributeDescriptor
            (VertexAttribute.TexCoord1 , VertexAttributeFormat.Float32, 2)
        );
        mesh.SetVertexBufferData(vertices, 0, 0, vcount);

        mesh.SetIndexBufferParams(vcount, IndexFormat.UInt32);
        mesh.SetIndexBufferData(indices, 0, 0, vcount);

        mesh.SetSubMesh(0, new SubMeshDescriptor(0, vcount));
    }

    public static void UpdateWithVertexArray
      (Mesh mesh, NativeArray<Vertex> vertices)
    {
        using (var indices = IndexArrayBuilder.SimpleArray(vertices.Length))
          MeshUtil.UpdateWithVertexIndexArrays(mesh, vertices, indices);
    }
}

#endregion

}
